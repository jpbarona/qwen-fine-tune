"""The real Qwen generator and GRPO trainer.

This module is the concrete `Generator` for GPU runs. It loads
`Qwen/Qwen3-0.6B` with thinking enabled, wraps it in a LoRA adapter over the
attention projections, and implements one GRPO update per group of sampled
rollouts.

Torch, transformers, and peft are imported lazily inside methods. The CPU test
suite runs without any of them. The pure helpers at the top (code extraction,
thinking split, group-normalized advantage, the clipped surrogate, and the
entropy formula) have no such dependency and are unit tested.

The entropy value on every generation is the mean Shannon entropy in nats of the
model's per-token output distribution over the generated tokens only. It is
computed from the per-step logits `generate` already asks for, never from a
proxy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .experiment import Generation, Prompt

__all__ = [
    "ATTENTION_TARGET_MODULES",
    "QwenGenerator",
    "categorical_entropy",
    "clipped_surrogate",
    "extract_code_from_answer",
    "group_advantages",
    "split_thinking_and_code",
]

#: LoRA is applied to the self-attention projections only, per the spec.
ATTENTION_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

#: Token id of `</think>` in the Qwen3 tokenizer. It separates the thinking
#: trace from the final answer. See the Qwen3 model card parsing example.
THINK_END_TOKEN_ID = 151668

#: The optimizer state and step are stored next to the adapter weights.
OPTIMIZER_STATE_NAME = "optimizer.pt"

#: A fenced Python block, with or without a language tag.
_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)


def extract_code_from_answer(text: str) -> str:
    """Extract Python code from a final answer, preferring a fenced block.

    A fenced block wins. Otherwise the whole answer is returned, stripped. The
    prompt already asks for code only, so an unfenced answer is kept as-is.
    """
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def split_thinking_and_code(text: str) -> tuple[str, str]:
    """Split a Qwen3 thinking response into (reasoning, answer) at `</think>`.

    A response with no `</think>` is treated as an empty trace with the whole
    text as the answer, so a non-thinking output still yields usable code.
    """
    marker = "</think>"
    index = text.rfind(marker)
    if index == -1:
        return "", text.strip()
    thinking = text[: index + len(marker)]
    answer = text[index + len(marker) :]
    thinking = thinking.replace("<think>", "").replace(marker, "").strip()
    return thinking, answer.strip()


def categorical_entropy(logits: Sequence[float]) -> float:
    """Shannon entropy in nats of a categorical distribution over `logits`.

    This is the pure reference for the torch computation in `generate`. For a
    uniform distribution over two values it returns ln(2), about 0.6931.
    """
    values = list(logits)
    if not values:
        return 0.0
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    probs = [exponent / total for exponent in exps]
    return -sum(prob * math.log(prob) for prob in probs if prob > 0.0)


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Group-normalized advantages: (r - mean) / std, zero when std is ~0.

    A group whose rewards are all equal has no relative signal, so every
    advantage is zero and the update has no gradient. This is the standard GRPO
    normalization and matches what the report describes.
    """
    values = list(rewards)
    if not values:
        raise ValueError("group_advantages needs at least one reward")
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0] * count
    return [(value - mean) / std for value in values]


def clipped_surrogate(ratio: float, advantage: float, epsilon: float) -> float:
    """The clipped policy-gradient term: min(ratio*A, clip(ratio, 1+-eps)*A).

    Pure reference for the torch elementwise version. Positive advantages want
    a larger ratio, negative advantages a smaller one, each bounded by the clip.
    """
    unclipped = ratio * advantage
    clipped = max(min(ratio, 1.0 + epsilon), 1.0 - epsilon) * advantage
    return min(unclipped, clipped)


@dataclass
class _GroupEntry:
    """One sampled completion kept for the next GRPO update.

    `input_ids` is the full prompt plus generated tokens. `old_logprob` is the
    policy logprob of the generated tokens at sampling time, detached. The
    reference logprob is only computed when the KL coefficient is positive.
    """

    input_ids: list[int]
    prompt_length: int
    old_logprob: float
    reference_logprob: float | None


class QwenGenerator:
    """The real generator and trainer behind the `Generator` seam.

    The class loads the model, tokenizer, LoRA adapter, and optimizer on
    construction. `generate` samples a group and caches the per-token logits it
    needs for the update. `update` runs one GRPO step over the last group.
    `save_checkpoint` and `load_checkpoint` write and restore the adapter
    weights, the optimizer state, and the step number.
    """

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
        max_prompt_tokens: int,
        max_completion_tokens: int,
        clip_epsilon: float,
        kl_beta: float,
        seed: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = float(learning_rate)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.clip_epsilon = float(clip_epsilon)
        self.kl_beta = float(kl_beta)
        self.seed = seed
        self.step = 0
        self._last_group: list[_GroupEntry] | None = None

        import torch  # lazy

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        from peft import LoraConfig, TaskType, get_peft_model  # lazy

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(ATTENTION_TARGET_MODULES),
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.train()

        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)

    # --- generation ------------------------------------------------------

    def generate(self, prompt: Prompt, *, n: int, temperature: float) -> list[Generation]:
        """Sample `n` generations for one prompt, splitting thinking from code."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        torch = self._torch
        do_sample = temperature > 0.0

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.problem}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
        prompt_ids = inputs.input_ids[0].tolist()
        prompt_length = len(prompt_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_completion_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generations: list[Generation] = []
        group: list[_GroupEntry] = []
        for index in range(n):
            generated_ids = outputs.sequences[index][prompt_length:].tolist()
            # Drop padding after the first end-of-sequence token so token counts
            # and entropy reflect the real completion, not the padded tail.
            eos = self.tokenizer.eos_token_id
            if eos is not None and eos in generated_ids:
                generated_ids = generated_ids[: generated_ids.index(eos) + 1]
            split = self._split_at_think_end(generated_ids)
            reasoning = self.tokenizer.decode(
                generated_ids[: split], skip_special_tokens=True
            ).strip()
            answer = self.tokenizer.decode(
                generated_ids[split:], skip_special_tokens=True
            ).strip()
            code = extract_code_from_answer(answer)

            scores = outputs.scores
            generated_length = min(len(generated_ids), len(scores))
            logits = torch.stack([scores[t][index] for t in range(generated_length)])

            mean_entropy = self._mean_entropy(logits).item()
            old_logprob = self._sequence_logprob(logits, generated_ids[:generated_length])
            reference_logprob = self._reference_logprob(
                prompt_ids, generated_ids[:generated_length]
            )

            generations.append(
                Generation(
                    reasoning=reasoning,
                    code=code,
                    prompt_tokens=prompt_length,
                    completion_tokens=generated_length,
                    mean_entropy=mean_entropy,
                )
            )
            group.append(
                _GroupEntry(
                    input_ids=prompt_ids + generated_ids[:generated_length],
                    prompt_length=prompt_length,
                    old_logprob=float(old_logprob),
                    reference_logprob=reference_logprob,
                )
            )

        self._last_group = group
        return generations

    def _split_at_think_end(self, generated_ids: list[int]) -> int:
        """The index just past the last `</think>` token, or 0 when absent."""
        for index in range(len(generated_ids) - 1, -1, -1):
            if generated_ids[index] == THINK_END_TOKEN_ID:
                return index + 1
        return 0

    @staticmethod
    def _mean_entropy(logits: Any) -> Any:
        """Mean per-token entropy in nats over a batch of logits."""
        import torch  # lazy

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()

    def _sequence_logprob(self, logits: Any, token_ids: list[int]) -> float:
        """The summed logprob of `token_ids` under `logits` (one per position)."""
        import torch  # lazy

        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids_t = torch.tensor(token_ids, device=logits.device)
        positions = torch.arange(len(token_ids), device=logits.device)
        chosen = log_probs[positions, token_ids_t]
        return float(chosen.sum().item())

    def _reference_logprob(self, prompt_ids: list[int], token_ids: list[int]) -> float | None:
        """The base-model logprob of the completion, or None when KL is off."""
        if self.kl_beta <= 0.0:
            return None
        import torch  # lazy

        sequence = prompt_ids + token_ids
        inputs = torch.tensor([sequence], device=self._device)
        with torch.no_grad():
            with self.model.disable_adapter():
                logits = self.model(inputs).logits[0]
        # log_probs[i] predicts the token at position i+1, so the completion
        # token at position prompt_length is predicted by log_probs[prompt_length - 1].
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        token_ids_t = torch.tensor(token_ids, device=self._device)
        positions = torch.arange(
            len(prompt_ids) - 1, len(prompt_ids) - 1 + len(token_ids), device=self._device
        )
        chosen = log_probs[positions, token_ids_t]
        return float(chosen.sum().item())

    # --- the GRPO update -------------------------------------------------

    def update(
        self,
        prompt: Prompt,
        generations: Sequence[Generation],
        rewards: Sequence[float],
    ) -> None:
        """One GRPO step over the last sampled group.

        The advantage is group-normalized. The ratio is clipped, and a KL term
        against the frozen base model is added when `kl_beta` is positive. The
        prompt and generations are accepted for interface symmetry with the
        seam; the token data comes from the group cached by `generate`.
        """
        del prompt, generations
        if self._last_group is None:
            raise RuntimeError("update called with no sampled group")
        if len(self._last_group) != len(rewards):
            raise ValueError(
                f"update got {len(rewards)} rewards for {len(self._last_group)} generations"
            )

        torch = self._torch
        advantages = group_advantages(list(rewards))
        advantages_tensor = torch.tensor(advantages, device=self._device)

        self.optimizer.zero_grad()
        losses: list[Any] = []
        for entry, advantage in zip(self._last_group, advantages_tensor):
            loss = self._group_loss(entry, advantage)
            losses.append(loss)
        total = torch.stack(losses).mean()
        total.backward()
        self.optimizer.step()
        self.step += 1
        self._last_group = None

    def _group_loss(self, entry: _GroupEntry, advantage: Any) -> Any:
        """The per-generation clipped surrogate loss plus optional KL."""
        torch = self._torch
        inputs = torch.tensor([entry.input_ids], device=self._device)
        logits = self.model(inputs).logits[0]
        token_ids = entry.input_ids[entry.prompt_length :]
        # log_probs[i] predicts position i+1, so the completion starts at index
        # prompt_length - 1 in the shifted distribution.
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        token_ids_t = torch.tensor(token_ids, device=self._device)
        positions = torch.arange(
            entry.prompt_length - 1,
            entry.prompt_length - 1 + len(token_ids),
            device=self._device,
        )
        new_logprob = log_probs[positions, token_ids_t].sum()

        ratio = torch.exp(new_logprob - entry.old_logprob)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        surrogate = torch.min(ratio * advantage, ratio_clipped * advantage)
        per_token = -surrogate

        if self.kl_beta > 0.0 and entry.reference_logprob is not None:
            divergence = entry.reference_logprob - new_logprob
            kl = torch.exp(divergence) - divergence - 1.0
            per_token = per_token + self.kl_beta * kl

        return per_token.mean()

    # --- checkpoint save and load ----------------------------------------

    def save_checkpoint(self, directory: str | Path) -> None:
        """Write the LoRA adapter and the optimizer state into `directory`."""
        import torch  # lazy

        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(folder))
        torch.save(
            {"step": self.step, "optimizer": self.optimizer.state_dict()},
            folder / OPTIMIZER_STATE_NAME,
        )

    def load_checkpoint(self, directory: str | Path) -> None:
        """Restore the adapter weights, optimizer state, and step number."""
        import torch  # lazy
        from peft import PeftModel  # lazy
        from transformers import AutoModelForCausalLM  # lazy

        folder = Path(directory)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.model = PeftModel.from_pretrained(self.base_model, str(folder))
        self.model.train()

        state = torch.load(folder / OPTIMIZER_STATE_NAME, map_location="cpu")
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["step"])
