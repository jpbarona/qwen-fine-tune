"""The shared Qwen3 and LFM2 causal language model GRPO trainer.

This module is the concrete `Generator` for GPU runs. It loads
the configured model with thinking enabled, wraps it in a LoRA adapter over the
attention projections, and implements one GRPO weight update per training step.
The vLLM engine samples the prompt groups, this module builds one group per
prompt and accumulates the gradient of every group, and the optimizer steps
once at the end. Advantage
is normalised inside a prompt group, never across the step. The backward pass
runs once per generation, so only one autograd graph is alive at a time and the
gradients accumulate across the group. That is what makes the 2,048 token cap
fit beside a larger prompt batch.

Prompts in a batch are left padded to one length, with an attention mask that
marks the pads. Every completion therefore starts at one fixed column, and the
generated rows split back into per prompt groups by index.

Torch, transformers, and peft are imported lazily inside methods. The CPU test
suite runs without any of them. The pure helpers at the top (code extraction,
thinking split, group-normalized advantage, the clipped surrogate, the entropy
formula, the entropy position picker, the batch padding and the row split) have
no such dependency and are unit tested.

Generation does not ask for per-step scores and keeps no logits. Entropy is
measured on the update pass, from the same forward logits the policy loss uses,
and only on the steps the caller marks. The value is the mean Shannon entropy in
nats over the generated tokens, estimated from at most `ENTROPY_MAX_TOKENS`
evenly spaced token positions when a completion is long. It is the entropy of
the policy distribution at temperature 1.0, which is the distribution the loss
sees. Entropy is logged, not rewarded, and the update pass stores nothing for it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .values import Generation, Prompt
from .model_profiles import LORA_TARGETS, model_profile, validate_transformers_version

__all__ = [
    "ATTENTION_TARGET_MODULES",
    "QwenGenerator",
    "CausalLMGenerator",
    "categorical_entropy",
    "check_attention_mask",
    "clipped_surrogate",
    "completion_logprobs",
    "completion_truncated",
    "entropy_positions",
    "extract_code_from_answer",
    "gradient_norm",
    "group_advantages",
    "k3_divergence",
    "log_softmax",
    "mean_token_loss",
    "pad_prompts",
    "seed_rngs",
    "split_thinking_and_code",
    "thinking_marker_index",
    "truncate_prompt",
    "update_norm",
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

#: At most this many generated token positions feed one entropy measurement.
#: Entropy is a diagnostic, so a bounded sample keeps the update pass cheap.
ENTROPY_MAX_TOKENS = 64

#: Row count per entropy slice. Bounds the float copy at about 10 MB a slice for
#: a 151k vocabulary.
ENTROPY_ROW_CHUNK = 16


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


def thinking_marker_index(token_ids: Sequence[int], marker_id: int) -> int:
    """The index just past the last marker token, or 0 when the marker is absent.

    This is the split point between the reasoning trace and the answer. A
    sequence with no marker has no reasoning block, so the whole thing is the
    answer and the index is 0. A repeated marker uses the last one, so a model
    that restates the closing marker still splits at the final copy.
    """
    for index in range(len(token_ids) - 1, -1, -1):
        if token_ids[index] == marker_id:
            return index + 1
    return 0


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


def entropy_positions(
    token_count: int, max_tokens: int = ENTROPY_MAX_TOKENS
) -> list[int]:
    """The generated token offsets used for one entropy measurement.

    Returns every offset when the completion is short enough. Otherwise it
    returns `max_tokens` evenly spaced offsets, including the first and the
    last. The mean over these offsets estimates the mean over the completion
    without reading every logit row.
    """
    if token_count < 0:
        raise ValueError(f"token_count must not be negative, got {token_count}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    if token_count == 0:
        return []
    if token_count <= max_tokens:
        return list(range(token_count))
    if max_tokens == 1:
        return [0]
    span = token_count - 1
    return [round(index * span / (max_tokens - 1)) for index in range(max_tokens)]


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


def mean_token_loss(per_token_losses: Sequence[Sequence[float]]) -> float:
    """Mean per-token loss over a group of completions.

    Every per-token loss is summed and the total is divided once by the total
    token count across the group. A token in a short completion weighs the same
    as a token in a long one, so the normalization is per token, never per
    sequence. This is the pure reference for the torch loss normalization.
    """
    total = 0.0
    count = 0
    for sequence in per_token_losses:
        total += sum(sequence)
        count += len(sequence)
    if count == 0:
        return 0.0
    return total / count


def gradient_norm(parameters: Iterable[Any]) -> float | None:
    """The total L2 norm of the gradients on the parameters that train.

    A parameter that does not require grad, or whose gradient is null, adds
    nothing. Each tensor's own 2-norm is exact, and `math.hypot` combines them
    without overflowing on a large model. A non-finite result is returned as
    None, so a diverged step leaves a metric row the JSON writer can write.
    """
    norm = 0.0
    for parameter in parameters:
        if not getattr(parameter, "requires_grad", False):
            continue
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        norm = math.hypot(norm, float(gradient.detach().norm(2)))
    return norm if math.isfinite(norm) else None


def update_norm(changes: Iterable[tuple[Any, Any]]) -> float | None:
    """The total L2 norm of the weight change made by one optimizer step.

    Each pair is a parameter after the step and a snapshot taken before it.
    The delta is the elementwise difference, and its 2-norm is combined with
    `math.hypot`. A non-finite result is returned as None, matching the
    gradient norm, so a diverged step still writes a readable metric row.
    """
    norm = 0.0
    for parameter, before in changes:
        delta = parameter.detach() - before
        norm = math.hypot(norm, float(delta.norm(2)))
    return norm if math.isfinite(norm) else None


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


def pad_prompts(
    prompt_ids: Sequence[Sequence[int]], pad_token_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad prompts to one length, with an attention mask for each row.

    Left padding puts every completion at the same column, so one fixed prefix
    length slices every generated row. A mask entry is 1 for a real token and 0
    for a pad. The mask is what tells the model not to attend to padding, so it
    is built here and checked by `check_attention_mask`.

    The tokenizer's own `padding_side` is not used. The batch is padded here so
    the side is fixed and the mask is built with it.
    """
    sequences = [list(ids) for ids in prompt_ids]
    if not sequences:
        raise ValueError("pad_prompts needs at least one prompt")
    for position, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError(f"prompt {position} holds no tokens")
    width = max(len(sequence) for sequence in sequences)
    padded = [
        [pad_token_id] * (width - len(sequence)) + sequence for sequence in sequences
    ]
    mask = [
        [0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences
    ]
    return padded, mask


def check_attention_mask(
    padded_ids: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]],
    pad_token_id: int,
) -> None:
    """Check that the padding is left aligned and masked.

    Every column the mask marks as padding must hold the pad token, every row
    must keep at least one real token, and the real tokens must sit in one block
    at the right end. A mask that exposes a pad is a batch builder bug, so it
    raises here instead of reaching the model.

    A real token that equals the pad token cannot be told apart when the pad
    token is also the end token, so that case is not rejected.
    """
    if len(padded_ids) != len(attention_mask):
        raise ValueError(
            f"got {len(padded_ids)} rows of tokens and {len(attention_mask)} mask rows"
        )
    for row, (ids, mask) in enumerate(zip(padded_ids, attention_mask)):
        if len(ids) != len(mask):
            raise ValueError(
                f"row {row} holds {len(ids)} tokens and {len(mask)} mask entries"
            )
        seen_real = False
        for column, (token, flag) in enumerate(zip(ids, mask)):
            if flag not in (0, 1):
                raise ValueError(f"row {row} column {column} has mask {flag}, expected 0 or 1")
            if flag == 0:
                if seen_real:
                    raise ValueError(f"row {row} is not left padded")
                if token != pad_token_id:
                    raise ValueError(f"row {row} masks a real token at column {column}")
            else:
                seen_real = True
        if not seen_real:
            raise ValueError(f"row {row} masks every token")


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

    `input_ids` is the prompt plus generated tokens. Qwen entries keep the
    rollout batch's leading padding and its attention mask. LFM entries remove
    leading padding so single-sequence convolution forwards match the masked
    rollout batch. `prompt_length` is the width of the stored prompt, where the
    completion starts. `old_logprobs` holds the per-token policy
    logprobs of the completion at temperature 1.0, captured at generation time
    and detached. `reference_logprobs` is the same under the base model with the
    adapter disabled, present only when the KL coefficient is positive.

    `prompt_id` guards the group order: `accumulate` checks it before using the
    entry, so a group can never be updated against another prompt's rewards.
    """

    prompt_id: str
    input_ids: list[int]
    attention_mask: list[int]
    prompt_length: int
    old_logprobs: list[float]
    reference_logprobs: list[float] | None


class QwenGenerator:
    """The real trainer behind the `Generator` seam. The engine samples.

    The class loads the model, tokenizer, LoRA adapter, and optimizer on
    construction. The vLLM engine samples token ids and
    `build_generation_batch` turns them into groups, caching the token ids,
    masks, and old logprobs the update needs, never the logits. `accumulate`
    runs the backward pass for one group and `apply_update` steps the optimizer
    once over every accumulated group. `update` keeps the single group
    shortcut. `save_checkpoint` and `load_checkpoint` write and restore the
    adapter weights, the optimizer state, and the step number, which counts
    weight updates.

    With `trainable` false the class loads the base model only. It builds no
    adapter and no optimizer, so a measurement run cannot train or checkpoint.
    That is the difficulty screen. Training keeps the default.
    """

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        lora_rank: int,
        lora_alpha: int,
        lora_targets: str = "attention",
        learning_rate: float,
        max_prompt_tokens: int,
        max_completion_tokens: int,
        clip_epsilon: float,
        kl_beta: float,
        seed: int | None = None,
        trainable: bool = True,
    ) -> None:
        if lora_targets not in LORA_TARGETS:
            raise ValueError(
                f"lora_targets must be one of {', '.join(LORA_TARGETS)}, "
                f"got {lora_targets!r}"
            )
        self.model_name = model_name
        self.revision = revision
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_targets = lora_targets
        self.learning_rate = float(learning_rate)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.clip_epsilon = float(clip_epsilon)
        self.kl_beta = float(kl_beta)
        self.seed = seed
        self.trainable = trainable
        self.profile = model_profile(model_name)
        self.step = 0
        self.last_grad_norm: float | None = None
        self.last_update_norm: float | None = None
        self._pending_groups: list[list[_GroupEntry]] | None = None
        self._accumulated_groups = 0

        if seed is not None:
            seed_rngs(seed)

        import torch  # lazy

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import transformers  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        validate_transformers_version(self.profile, transformers.__version__)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        marker_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(marker_ids) != 1 or marker_ids[0] == self.tokenizer.unk_token_id:
            raise ValueError("the model tokenizer must encode </think> as one known token")
        self.think_end_token_id = marker_ids[0]

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.profile = model_profile(
            model_name, model_type=self.base_model.config.model_type
        )
        validate_transformers_version(self.profile, transformers.__version__)

        if not trainable:
            self.model = self.base_model
            self.model.eval()
            self.optimizer = None
            return

        from peft import LoraConfig, TaskType, get_peft_model  # lazy

        target_modules: Any = (
            "all-linear"
            if lora_targets == "all-linear"
            else list(self.profile.attention_targets)
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.lora_target_modules = tuple(
            self.model.peft_config["default"].target_modules
        )
        self.model.train()

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

    # --- prompt encoding -------------------------------------------------

    def encode_prompt(
        self, prompt: Prompt, *, thinking: bool = True
    ) -> tuple[list[int], bool]:
        """Prompt token ids and the truncation flag, for an external sampler.

        The vLLM backend cannot tokenize, so the trainer encodes the prompt and
        the engine samples from these ids. The prompt cap is applied here.
        """
        return self._encode_prompt(prompt, thinking=thinking)

    def _encode_prompt(
        self, prompt: Prompt, *, thinking: bool
    ) -> tuple[list[int], bool]:
        """Chat template, tokenize, and truncate one prompt to the token cap."""
        template_options = {}
        if self.profile.supports_thinking_toggle:
            template_options["enable_thinking"] = thinking
        # The template owns BOS and other control tokens. Tokenizing its text a
        # second time with add_special_tokens=True duplicates the LFM BOS.
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.problem}],
            tokenize=True,
            add_generation_prompt=True,
            **template_options,
        )
        raw_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        return truncate_prompt(raw_ids, self.max_prompt_tokens)

    def build_generation_batch(
        self,
        prompts: Sequence[Prompt],
        prompt_ids: Sequence[Sequence[int]],
        prompt_truncated: Sequence[bool],
        completion_token_ids: Sequence[Sequence[Sequence[int]]],
        *,
        cache: bool,
    ) -> list[list[Generation]]:
        """Build generation objects from token ids sampled by the vLLM engine.

        Each prompt is padded on its own, so the mask is all ones and the
        completion starts at the prompt length. When `cache` is true the old
        logprobs are computed on the training model and the group data is
        cached for the update pass.
        """
        prompt_list = list(prompts)
        counts = (
            len(prompt_list),
            len(prompt_ids),
            len(prompt_truncated),
            len(completion_token_ids),
        )
        if len(set(counts)) != 1:
            raise ValueError(
                f"build_generation_batch got mismatched lengths {counts}: "
                "prompts, prompt ids, truncation flags and completion rows must match"
            )
        if cache and not self.trainable:
            raise RuntimeError(
                "this generator was built without an adapter and cannot cache groups"
            )
        if cache:
            # A new sample always starts a fresh step, including after an
            # interrupted first group's backward pass.
            self.optimizer.zero_grad()
            self._accumulated_groups = 0
            self._pending_groups = None
        if cache:
            self.model.train()
        else:
            self.model.eval()

        pad_token_id = self.tokenizer.pad_token_id
        generations: list[list[Generation]] = []
        cached_groups: list[list[_GroupEntry]] = []
        for index, prompt in enumerate(prompt_list):
            padded, mask = pad_prompts([prompt_ids[index]], pad_token_id)
            check_attention_mask(padded, mask, pad_token_id)
            group, entries = self._decode_group(
                prompt,
                padded[0],
                mask[0],
                len(padded[0]),
                prompt_truncated=prompt_truncated[index],
                completion_rows=completion_token_ids[index],
                caching=cache,
            )
            generations.append(group)
            cached_groups.append(entries)

        if cache:
            self._pending_groups = cached_groups
        return generations

    def _decode_group(
        self,
        prompt: Prompt,
        padded_ids: Sequence[int],
        attention_mask: Sequence[int],
        prompt_length: int,
        *,
        prompt_truncated: bool,
        completion_rows: Sequence[Sequence[int]],
        caching: bool,
    ) -> tuple[list[Generation], list[_GroupEntry]]:
        """Decode one prompt's completion rows and cache its update data.

        The vLLM batch builder passes the engine's token ids here, so every
        group gets the same generation objects and group entries.
        """
        torch = self._torch
        eos = self.tokenizer.eos_token_id
        real_prompt_length = sum(attention_mask)
        # Transformers 5.0 LFM convolution masks padding only for batch sizes
        # above one. Teacher forcing runs one sequence at a time, so retaining
        # a rollout batch's left pads would change its convolution states.
        cache_offset = (
            prompt_length - real_prompt_length
            if self.profile.model_type == "lfm2"
            else 0
        )
        cache_prompt_length = prompt_length - cache_offset
        group: list[Generation] = []
        entries: list[_GroupEntry] = []
        for raw_ids in completion_rows:
            generated_ids = [int(token) for token in raw_ids]
            truncated = completion_truncated(generated_ids, eos)
            if not truncated:
                generated_ids = generated_ids[: generated_ids.index(eos) + 1]
            split = self._split_at_think_end(generated_ids)
            reasoning = self.tokenizer.decode(
                generated_ids[:split], skip_special_tokens=True
            ).strip()
            answer = self.tokenizer.decode(
                generated_ids[split:], skip_special_tokens=True
            ).strip()
            code = extract_code_from_answer(answer)

            group.append(
                Generation(
                    reasoning=reasoning,
                    code=code,
                    prompt_tokens=real_prompt_length,
                    completion_tokens=len(generated_ids),
                    truncated=truncated,
                    prompt_truncated=prompt_truncated,
                    closed=split > 0,
                )
            )

            if caching:
                entry_ids = list(padded_ids)[cache_offset:] + generated_ids
                entry_mask = list(attention_mask)[cache_offset:] + [1] * len(generated_ids)
                with torch.no_grad():
                    sequence = torch.tensor([entry_ids], device=self._device)
                    old_logprobs = self._completion_logprobs(
                        sequence, cache_prompt_length, entry_mask
                    ).tolist()
                reference_logprobs = self._reference_logprobs(
                    entry_ids, cache_prompt_length, entry_mask, generated_ids
                )
                entries.append(
                    _GroupEntry(
                        prompt_id=prompt.prompt_id,
                        input_ids=entry_ids,
                        attention_mask=entry_mask,
                        prompt_length=cache_prompt_length,
                        old_logprobs=old_logprobs,
                        reference_logprobs=reference_logprobs,
                    )
                )
        return group, entries

    def _split_at_think_end(self, generated_ids: list[int]) -> int:
        """The index just past the last closing reasoning marker, or 0 when absent."""
        return thinking_marker_index(generated_ids, self.think_end_token_id)

    def _completion_logprobs(
        self, sequence: Any, prompt_length: int, attention_mask: list[int] | None = None
    ) -> Any:
        """Per-token logprobs of the completion under the policy at temperature 1.0.

        This is the only path used for both the old and the new logprobs, so the
        ratio is 1.0 on the first update when the policy has not changed. Qwen
        keeps the rollout mask, and LFM uses an unpadded sequence to preserve
        the masked rollout batch's convolution states.
        """
        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = self._torch.tensor(
                [attention_mask], dtype=self._torch.long, device=sequence.device
            )
        logits = self.model(sequence, use_cache=False, **kwargs).logits[0]
        token_ids = sequence[0][prompt_length:].tolist()
        return self._completion_logprobs_from_logits(logits, prompt_length, token_ids)

    def _completion_logprobs_from_logits(
        self, logits: Any, prompt_length: int, token_ids: Sequence[int]
    ) -> Any:
        """Completion logprobs from logits the update pass already computed."""
        torch = self._torch
        start = prompt_length - 1
        completion_logits = logits[start : start + len(token_ids)]
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        ids = torch.tensor(list(token_ids), device=logits.device)
        positions = torch.arange(len(token_ids), device=logits.device)
        return log_probs[positions, ids]

    def _sampled_entropy(
        self, logits: Any, prompt_length: int, token_count: int
    ) -> tuple[float, int]:
        """Summed entropy in nats over a bounded sample of generated positions.

        Runs under `no_grad` on the logits the loss already holds, so nothing is
        stored and no gradient flows. Returns the sum and the position count,
        so the group mean can be taken over the sampled positions in `accumulate`.
        """
        torch = self._torch
        offsets = entropy_positions(token_count)
        if not offsets:
            return 0.0, 0
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(offsets), ENTROPY_ROW_CHUNK):
                chunk = offsets[start : start + ENTROPY_ROW_CHUNK]
                rows = torch.tensor(
                    [prompt_length - 1 + offset for offset in chunk],
                    device=logits.device,
                )
                log_probs = torch.log_softmax(logits[rows].float(), dim=-1)
                probs = log_probs.exp()
                terms = torch.where(
                    probs > 0, probs * log_probs, torch.zeros_like(probs)
                )
                total += float((-terms.sum(dim=-1)).sum().item())
                count += len(chunk)
        if not math.isfinite(total):
            raise RuntimeError("mean token entropy is not finite")
        return total, count

    def _reference_logprobs(
        self,
        input_ids: list[int],
        prompt_length: int,
        attention_mask: list[int],
        token_ids: list[int],
    ) -> list[float] | None:
        """The per-token base-model logprob of the completion, or None when KL is off.

        The input is the padded prompt plus the completion, with the same mask
        the sampling call used, so the pads stay out of the attention.
        """
        if self.kl_beta <= 0.0:
            return None
        import torch  # lazy

        inputs = torch.tensor([input_ids], device=self._device)
        mask = torch.tensor(
            [attention_mask], dtype=torch.long, device=self._device
        )
        with torch.no_grad():
            with self.model.disable_adapter():
                logits = self.model(inputs, attention_mask=mask, use_cache=False).logits[0]
        return self._completion_logprobs_from_logits(
            logits, prompt_length, token_ids
        ).tolist()

    # --- the GRPO update -------------------------------------------------

    def accumulate(
        self,
        prompt: Prompt,
        generations: Sequence[Generation],
        rewards: Sequence[float],
        *,
        measure_entropy: bool = False,
    ) -> float | None:
        """Add the gradient of one prompt group to the pending step.

        The advantage is group-normalized, so the normalisation never crosses a
        prompt boundary. The per-token clipped surrogate and the per-token k3
        KL are summed over the whole group and divided once by the group's token
        count, so a token in a short completion weighs the same as a token in a
        long one. `apply_update` then divides the accumulated gradient by the
        group count, which makes the step the mean over the prompt batch.

        The prompt and generations are accepted for interface symmetry with the
        seam. The token data comes from the group cached by
        `build_generation_batch`, and the group is popped in order, so the
        rewards must belong to the prompt the entry was sampled for. That is
        checked.

        Backward runs once per generation and the gradients accumulate across
        the group. Only one autograd graph is alive at a time, which is what
        makes a larger batch fit at the 2,048 token cap.

        When `measure_entropy` is true, the forward pass that computes the new
        logprobs also yields a bounded per-token entropy sample. The group mean
        is returned and the caller writes it to the step metric. When false,
        nothing is measured and None is returned. A non-trainable generator has
        no adapter or optimizer and refuses to accumulate.
        """
        del generations
        if not self.trainable:
            raise RuntimeError(
                "this generator was built without an adapter and cannot update"
            )
        if self._pending_groups is None:
            raise RuntimeError("accumulate called with no sampled groups")
        if not self._pending_groups:
            raise RuntimeError("accumulate called more times than the sampler returned")
        group = self._pending_groups.pop(0)
        if group and group[0].prompt_id != prompt.prompt_id:
            raise ValueError(
                f"accumulate got prompt {prompt.prompt_id!r} for the group sampled "
                f"for {group[0].prompt_id!r}"
            )
        if len(group) != len(rewards):
            raise ValueError(
                f"accumulate got {len(rewards)} rewards for {len(group)} generations"
            )

        torch = self._torch
        self.model.train()

        advantages = group_advantages(list(rewards))
        advantages_tensor = torch.tensor(advantages, device=self._device)

        # The loss is divided once by the whole group's token count, so the
        # per-generation backward calls add up to the same gradient as one
        # summed backward would. The difference is peak memory: each graph is
        # freed before the next forward pass starts.
        token_count = sum(len(entry.old_logprobs) for entry in group)
        if token_count == 0:
            raise RuntimeError("accumulate got a group with no completion tokens")

        if self._accumulated_groups == 0:
            # A fresh step starts with clean gradients. A step that failed
            # halfway leaves some behind, so zero on the first group, not on
            # every group.
            self.optimizer.zero_grad()
        entropy_total = 0.0
        entropy_tokens = 0
        for entry, advantage in zip(group, advantages_tensor):
            surrogate_sum, kl_sum, entropy_sum, measured = self._group_terms(
                entry, advantage, measure_entropy
            )
            loss = -surrogate_sum / token_count
            if kl_sum is not None:
                loss = loss + self.kl_beta * kl_sum / token_count
            loss.backward()
            if entropy_sum is not None:
                entropy_total += entropy_sum
                entropy_tokens += measured

        self._accumulated_groups += 1
        if measure_entropy and entropy_tokens:
            return entropy_total / entropy_tokens
        return None

    def apply_update(self) -> float | None:
        """Step the optimizer once over every group accumulated since the last call.

        The accumulated gradient is divided by the group count first, so the
        update is the mean over the prompt batch instead of a sum that grows
        with the batch size. The total L2 norm of those mean gradients is
        stored on the instance and returned, so the step metric records what
        the optimizer was about to apply. AdamW hides how far the weights
        moved, so the trainable weights are snapshotted before the step and
        the norm of their change is stored as `last_update_norm`. Both norms
        are null when they are not finite. The weight update counter moves
        once per call, which is also the step number a checkpoint stores.
        """
        if not self.trainable:
            raise RuntimeError(
                "this generator was built without an adapter and cannot update"
            )
        if self._accumulated_groups == 0:
            raise RuntimeError("apply_update called with no accumulated groups")

        count = self._accumulated_groups
        for parameter in self.model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.grad.div_(count)

        # AdamW normalises each step by the second moment of the gradient, so
        # the gradient norm stays close across learning rates. The change in
        # the weights is what shows that the rate reached them.
        before = [
            (parameter, parameter.detach().clone())
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        self.last_grad_norm = gradient_norm(self.model.parameters())
        self.optimizer.step()
        self.last_update_norm = update_norm(before)
        self.optimizer.zero_grad()
        self.step += 1
        self._accumulated_groups = 0
        self._pending_groups = None
        return self.last_grad_norm

    def update(
        self,
        prompt: Prompt,
        generations: Sequence[Generation],
        rewards: Sequence[float],
        *,
        measure_entropy: bool = False,
    ) -> float | None:
        """One group shortcut: accumulate the group, then apply the update.

        The training loop uses `accumulate` and `apply_update` separately so a
        whole step shares one optimizer step. This wrapper keeps the single
        group path, which is also the shape the fake generator exposes.
        """
        entropy = self.accumulate(
            prompt, generations, rewards, measure_entropy=measure_entropy
        )
        self.apply_update()
        return entropy

    def _group_terms(
        self, entry: _GroupEntry, advantage: Any, measure_entropy: bool
    ) -> tuple[Any, Any | None, float | None, int]:
        """Summed per-token surrogate and KL for one generation.

        Also returns the summed sampled entropy and its position count when the
        caller asked for a measurement, both None and zero otherwise. The stored
        attention mask is fed back so the pads never enter the attention.
        """
        torch = self._torch
        inputs = torch.tensor([entry.input_ids], device=self._device)
        mask = torch.tensor(
            [entry.attention_mask], dtype=torch.long, device=self._device
        )
        logits = self.model(inputs, attention_mask=mask, use_cache=False).logits[0]
        token_ids = entry.input_ids[entry.prompt_length :]
        new_logprobs = self._completion_logprobs_from_logits(
            logits, entry.prompt_length, token_ids
        )
        old_logprobs = torch.tensor(entry.old_logprobs, device=new_logprobs.device)

        ratio = torch.exp(new_logprobs - old_logprobs)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        surrogate = torch.min(ratio * advantage, ratio_clipped * advantage)
        surrogate_sum = surrogate.sum()

        kl_sum: Any | None = None
        if self.kl_beta > 0.0 and entry.reference_logprobs is not None:
            reference = torch.tensor(entry.reference_logprobs, device=new_logprobs.device)
            divergence = reference - new_logprobs
            kl_sum = (torch.exp(divergence) - divergence - 1.0).sum()

        entropy_sum: float | None = None
        entropy_count = 0
        if measure_entropy:
            entropy_sum, entropy_count = self._sampled_entropy(
                logits, entry.prompt_length, len(token_ids)
            )

        return surrogate_sum, kl_sum, entropy_sum, entropy_count

    # --- checkpoint save and load ----------------------------------------

    def save_checkpoint(self, directory: str | Path) -> None:
        """Write the LoRA adapter and the optimizer state into `directory`.

        The stored step is the number of applied weight updates, so a resume
        continues the same counter the training loop reports.
        """
        if not self.trainable:
            raise RuntimeError(
                "this generator was built without an adapter and has no checkpoint"
            )
        import torch  # lazy

        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(folder))
        torch.save(
            {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "model_name": self.model_name,
                "model_revision": self.revision,
            },
            folder / OPTIMIZER_STATE_NAME,
        )

    def save_lora_adapter(self, directory: str | Path) -> Path:
        """Write just the LoRA adapter, for the vLLM engine to load."""
        if not self.trainable:
            raise RuntimeError(
                "this generator was built without an adapter and has none to save"
            )
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(folder))
        return folder

    def load_checkpoint(self, directory: str | Path) -> None:
        """Restore the adapter weights, optimizer state, and step number."""
        if not self.trainable:
            raise RuntimeError(
                "this generator was built without an adapter and cannot load a checkpoint"
            )
        import torch  # lazy
        from peft import get_peft_model_state_dict, set_peft_model_state_dict  # lazy
        from peft.utils.save_and_load import load_peft_weights

        folder = Path(directory)
        state = torch.load(folder / OPTIMIZER_STATE_NAME, map_location="cpu")
        adapter_config = json.loads((folder / "adapter_config.json").read_text())
        self._validate_checkpoint_identity(adapter_config, state)
        weights = load_peft_weights(str(folder), device="cpu")
        expected_weights = get_peft_model_state_dict(self.model, adapter_name="default")
        missing = sorted(expected_weights.keys() - weights.keys())
        unexpected = sorted(weights.keys() - expected_weights.keys())
        if missing or unexpected:
            raise ValueError(
                f"checkpoint adapter tensor keys differ; missing={missing}, "
                f"unexpected={unexpected}"
            )
        for name, tensor in weights.items():
            if tensor.shape != expected_weights[name].shape:
                raise ValueError(f"checkpoint adapter tensor shape differs for {name!r}")
        # Reuse the already-loaded base and trainable adapter. Loading another
        # full base model temporarily doubles VRAM during every resume.
        set_peft_model_state_dict(self.model, weights, adapter_name="default")
        self.optimizer.load_state_dict(state["optimizer"])
        # The saved optimizer carries the rate it trained with. The config is
        # the authority on resume, so a run can be slowed or stopped by editing
        # the rate and resuming from the last checkpoint.
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self.optimizer.zero_grad()
        self.model.train()
        self.step = int(state["step"])
        self._pending_groups = None
        self._accumulated_groups = 0

    def _validate_checkpoint_identity(self, adapter_config: dict, state: dict) -> None:
        """Accept older checkpoints while rejecting a different model or LoRA shape."""
        base_name = adapter_config.get("base_model_name_or_path")
        if base_name and base_name != self.model_name:
            raise ValueError(
                f"checkpoint base model {base_name!r} differs from {self.model_name!r}"
            )
        expected = {
            "model_name": self.model_name,
            "model_revision": self.revision,
        }
        for field, value in expected.items():
            if field in state and state[field] != value:
                raise ValueError(f"checkpoint {field} differs from the configured model")
        for field, value in (("r", self.lora_rank), ("lora_alpha", self.lora_alpha)):
            if adapter_config.get(field) != value:
                raise ValueError(f"checkpoint adapter {field} differs from the configured LoRA")
        if set(adapter_config.get("target_modules", [])) != set(self.lora_target_modules):
            raise ValueError(
                "checkpoint LoRA target modules differ from the configured placement"
            )


# Retain the established import and class name for callers and old integrations.
CausalLMGenerator = QwenGenerator
