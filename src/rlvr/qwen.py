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
    "completion_logprobs",
    "completion_truncated",
    "extract_code_from_answer",
    "group_advantages",
    "k3_divergence",
    "log_softmax",
    "seed_rngs",
    "split_thinking_and_code",
    "truncate_prompt",
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


def log_softmax(logits: Sequence[float]) -> list[float]:
    """Log-softmax of a logits vector in nats, in a numerically stable way."""
    values = list(logits)
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [math.log(exponent / total) for exponent in exps]


def completion_logprobs(
    logits: Sequence[Sequence[float]],
    token_ids: Sequence[int],
    prompt_length: int,
) -> list[float]:
    """Per-token logprobs of a completion under per-position forward logits.

    `logits[i]` predicts the token at position `i + 1` of the full sequence. The
    completion tokens sit at positions `prompt_length` and onward, so the token
    `token_ids[t]` is predicted by `logits[prompt_length - 1 + t]`. This mirrors
    the torch computation in `_completion_logprobs` and is the pure reference
    used to prove the ratio is 1.0 when the policy has not changed.
    """
    result: list[float] = []
    for offset, token in enumerate(token_ids):
        position = prompt_length + offset
        result.append(log_softmax(logits[position - 1])[token])
    return result


def k3_divergence(
    reference_logprobs: Sequence[float],
    policy_logprobs: Sequence[float],
) -> float:
    """The per-token k3 KL estimator, summed over tokens.

    For one token the estimator is `exp(d) - d - 1` where `d` is the reference
    logprob minus the policy logprob. It is summed per token, never computed on
    the summed log-ratio, because the estimator is not linear.
    """
    total = 0.0
    for reference, policy in zip(reference_logprobs, policy_logprobs):
        divergence = reference - policy
        total += math.exp(divergence) - divergence - 1.0
    return total


def truncate_prompt(
    token_ids: Sequence[int], max_tokens: int
) -> tuple[list[int], bool]:
    """Truncate a prompt token sequence to `max_tokens`.

    Returns the kept ids and whether anything was dropped. The prompt length is
    capped at `max_tokens`, and the flag records the truncation for the metric.
    """
    ids = list(token_ids)
    if len(ids) > max_tokens:
        return ids[:max_tokens], True
    return ids, False


def completion_truncated(token_ids: Sequence[int], eos_token_id: int | None) -> bool:
    """Whether a completion hit the length cap without an end token."""
    ids = list(token_ids)
    return eos_token_id is None or eos_token_id not in ids


def seed_rngs(seed: int) -> None:
    """Seed the Python, numpy, and torch RNGs.

    torch and numpy are optional here so the CPU suite can call this without
    them installed. On a GPU host all three are seeded.
    """
    import random

    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class _GroupEntry:
    """One sampled completion kept for the next GRPO update.

    `input_ids` is the full prompt plus generated tokens. `old_logprobs` holds
    the per-token policy logprobs of the completion at temperature 1.0, captured
    at generation time and detached. `reference_logprobs` is the same under the
    base model with the adapter disabled, present only when the KL coefficient
    is positive.
    """

    input_ids: list[int]
    prompt_length: int
    old_logprobs: list[float]
    reference_logprobs: list[float] | None


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

        if seed is not None:
            seed_rngs(seed)

        import torch  # lazy

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._rng = torch.Generator(device=self._device)
        if seed is not None:
            self._rng.manual_seed(seed)

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
        raw_prompt_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        prompt_ids, prompt_truncated = truncate_prompt(raw_prompt_ids, self.max_prompt_tokens)
        prompt_length = len(prompt_ids)
        inputs = {
            "input_ids": torch.tensor([prompt_ids], device=self._device),
            "attention_mask": torch.ones(1, prompt_length, dtype=torch.long, device=self._device),
        }

        generate_kwargs: dict[str, Any] = dict(
            max_new_tokens=self.max_completion_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if do_sample:
            generate_kwargs["generator"] = self._rng

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        generations: list[Generation] = []
        group: list[_GroupEntry] = []
        for index in range(n):
            raw_ids = outputs.sequences[index][prompt_length:].tolist()
            eos = self.tokenizer.eos_token_id
            if eos is not None and eos in raw_ids:
                generated_ids = raw_ids[: raw_ids.index(eos) + 1]
                truncated = False
            else:
                generated_ids = raw_ids
                truncated = True
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

            with torch.no_grad():
                sequence = torch.tensor(
                    [prompt_ids + generated_ids[:generated_length]], device=self._device
                )
                old_logprobs = self._completion_logprobs(sequence, prompt_length).tolist()
            reference_logprobs = self._reference_logprobs(
                prompt_ids, generated_ids[:generated_length]
            )

            generations.append(
                Generation(
                    reasoning=reasoning,
                    code=code,
                    prompt_tokens=prompt_length,
                    completion_tokens=generated_length,
                    mean_entropy=mean_entropy,
                    truncated=truncated,
                    prompt_truncated=prompt_truncated,
                )
            )
            group.append(
                _GroupEntry(
                    input_ids=prompt_ids + generated_ids[:generated_length],
                    prompt_length=prompt_length,
                    old_logprobs=old_logprobs,
                    reference_logprobs=reference_logprobs,
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

    def _completion_logprobs(self, sequence: Any, prompt_length: int) -> Any:
        """Per-token logprobs of the completion under the policy at temperature 1.0.

        This is the only path used for both the old and the new logprobs, so the
        ratio is 1.0 on the first update when the policy has not changed. The
        generation scores are not used here: they are temperature-scaled.
        """
        torch = self._torch
        logits = self.model(sequence).logits[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        token_ids = sequence[0][prompt_length:].tolist()
        token_ids_t = torch.tensor(token_ids, device=logits.device)
        positions = torch.arange(
            prompt_length - 1, prompt_length - 1 + len(token_ids), device=logits.device
        )
        return log_probs[positions, token_ids_t]

    def _reference_logprobs(
        self, prompt_ids: list[int], token_ids: list[int]
    ) -> list[float] | None:
        """The per-token base-model logprob of the completion, or None when KL is off."""
        if self.kl_beta <= 0.0:
            return None
        import torch  # lazy

        sequence = prompt_ids + token_ids
        inputs = torch.tensor([sequence], device=self._device)
        with torch.no_grad():
            with self.model.disable_adapter():
                logits = self.model(inputs).logits[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        token_ids_t = torch.tensor(token_ids, device=logits.device)
        positions = torch.arange(
            len(prompt_ids) - 1, len(prompt_ids) - 1 + len(token_ids), device=logits.device
        )
        return log_probs[positions, token_ids_t].tolist()

    # --- the GRPO update -------------------------------------------------

    def update(
        self,
        prompt: Prompt,
        generations: Sequence[Generation],
        rewards: Sequence[float],
    ) -> None:
        """One GRPO step over the last sampled group.

        The advantage is group-normalized. The ratio is per-token and clipped,
        and a per-token k3 KL term against the frozen base model is added when
        `kl_beta` is positive. The prompt and generations are accepted for
        interface symmetry with the seam; the token data comes from the group
        cached by `generate`.
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
            losses.append(self._group_loss(entry, advantage))
        total = torch.stack(losses).mean()
        total.backward()
        self.optimizer.step()
        self.step += 1
        self._last_group = None

    def _group_loss(self, entry: _GroupEntry, advantage: Any) -> Any:
        """The per-token clipped surrogate loss plus optional per-token KL."""
        torch = self._torch
        inputs = torch.tensor([entry.input_ids], device=self._device)
        new_logprobs = self._completion_logprobs(inputs, entry.prompt_length)
        old_logprobs = torch.tensor(entry.old_logprobs, device=new_logprobs.device)

        ratio = torch.exp(new_logprobs - old_logprobs)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        surrogate = torch.min(ratio * advantage, ratio_clipped * advantage)
        loss = -surrogate.mean()

        if self.kl_beta > 0.0 and entry.reference_logprobs is not None:
            reference = torch.tensor(entry.reference_logprobs, device=new_logprobs.device)
            divergence = reference - new_logprobs
            kl = (torch.exp(divergence) - divergence - 1.0).mean()
            loss = loss + self.kl_beta * kl

        return loss

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
