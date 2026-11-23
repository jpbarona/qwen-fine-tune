"""Tests for the pure helpers in qwen.py. No torch is imported."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from rlvr.engine.qwen import (
    ATTENTION_TARGET_MODULES,
    QwenGenerator,
    categorical_entropy,
    check_attention_mask,
    clipped_surrogate,
    completion_logprobs,
    completion_truncated,
    entropy_positions,
    extract_code_from_answer,
    gradient_norm,
    group_advantages,
    k3_divergence,
    log_softmax,
    mean_token_loss,
    pad_prompts,
    seed_rngs,
    split_thinking_and_code,
    thinking_marker_index,
    truncate_prompt,
    update_norm,
)


def test_attention_target_modules_are_the_four_projections() -> None:
    assert ATTENTION_TARGET_MODULES == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_extract_code_prefers_a_fenced_python_block() -> None:
    text = "Here is the code:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code_from_answer(text) == "def f():\n    return 1"


def test_extract_code_accepts_a_fence_without_a_language() -> None:
    text = "```\ndef f():\n    return 2\n```"
    assert extract_code_from_answer(text) == "def f():\n    return 2"


def test_extract_code_returns_unfenced_text_stripped() -> None:
    text = "\n\ndef f():\n    return 3\n\n"
    assert extract_code_from_answer(text) == "def f():\n    return 3"


def test_split_thinking_keeps_reasoning_apart_from_the_answer() -> None:
    text = "<think>\nPlan the loop.\n</think>\n\ndef f():\n    return 1\n"
    reasoning, answer = split_thinking_and_code(text)
    assert reasoning == "Plan the loop."
    assert answer == "def f():\n    return 1"


def test_split_thinking_uses_the_last_marker() -> None:
    text = "<think>first</think>\n<think>second</think>\nanswer here"
    reasoning, answer = split_thinking_and_code(text)
    assert reasoning == "first\nsecond"
    assert answer == "answer here"


def test_split_thinking_without_a_marker_is_all_answer() -> None:
    reasoning, answer = split_thinking_and_code("def f():\n    return 1")
    assert reasoning == ""
    assert answer == "def f():\n    return 1"


def test_thinking_marker_index_returns_the_position_past_the_marker() -> None:
    assert thinking_marker_index([1, 2, 3], 2) == 2
    assert thinking_marker_index([2, 5, 7], 2) == 1
    assert thinking_marker_index([1, 2, 3], 3) == 3


def test_thinking_marker_index_is_zero_when_the_marker_is_absent() -> None:
    assert thinking_marker_index([1, 2, 3], 9) == 0
    assert thinking_marker_index([], 9) == 0


def test_thinking_marker_index_uses_the_last_marker() -> None:
    assert thinking_marker_index([2, 5, 2, 7], 2) == 3
    assert thinking_marker_index([2, 2], 2) == 2


def test_categorical_entropy_of_uniform_two_values_is_ln2() -> None:
    assert categorical_entropy([0.0, 0.0]) == pytest.approx(math.log(2))


def test_categorical_entropy_of_a_peaked_distribution_is_small() -> None:
    assert categorical_entropy([100.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)


def test_entropy_positions_uses_every_token_when_short() -> None:
    assert entropy_positions(4, max_tokens=8) == [0, 1, 2, 3]


def test_entropy_positions_spans_the_completion_when_long() -> None:
    offsets = entropy_positions(1000, max_tokens=5)
    assert offsets == [0, 250, 500, 749, 999]
    assert offsets == sorted(set(offsets))


def test_entropy_positions_handles_no_tokens_and_one_sample() -> None:
    assert entropy_positions(0) == []
    assert entropy_positions(1000, max_tokens=1) == [0]


def test_entropy_positions_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="token_count"):
        entropy_positions(-1)
    with pytest.raises(ValueError, match="max_tokens"):
        entropy_positions(4, max_tokens=0)


class _Gradient:
    """A stand-in for a tensor gradient with one 2-norm call."""

    def __init__(self, *values: float) -> None:
        self.values = values

    def detach(self) -> "_Gradient":
        return self

    def norm(self, order: int) -> float:
        assert order == 2
        return math.sqrt(sum(value * value for value in self.values))


class _Parameter:
    """A stand-in for a trainable parameter."""

    def __init__(self, grad, requires_grad: bool = True) -> None:
        self.grad = grad
        self.requires_grad = requires_grad


def test_gradient_norm_is_the_euclidean_norm_over_parameters() -> None:
    parameters = [
        _Parameter(_Gradient(3.0, 0.0)),
        _Parameter(_Gradient(0.0, 4.0)),
    ]
    assert gradient_norm(parameters) == pytest.approx(5.0)


def test_gradient_norm_ignores_a_parameter_with_a_null_gradient() -> None:
    parameters = [
        _Parameter(_Gradient(3.0, 0.0)),
        _Parameter(None),
        _Parameter(_Gradient(0.0, 4.0)),
    ]
    assert gradient_norm(parameters) == pytest.approx(5.0)


class _UpdateTensor:
    """A stand-in for a tensor with the calls the norms make."""

    def __init__(
        self, *values: float, label: str = "", events: list[str] | None = None
    ) -> None:
        self.values = tuple(float(value) for value in values)
        self.label = label
        self.events = events if events is not None else []

    def detach(self) -> "_UpdateTensor":
        return self

    def clone(self) -> "_UpdateTensor":
        return _UpdateTensor(*self.values, label=self.label, events=self.events)

    def div_(self, count: int) -> None:
        self.values = tuple(value / count for value in self.values)

    def __sub__(self, other: "_UpdateTensor") -> "_UpdateTensor":
        return _UpdateTensor(
            *(left - right for left, right in zip(self.values, other.values)),
            label="update",
            events=self.events,
        )

    def norm(self, order: int) -> float:
        assert order == 2
        value = math.sqrt(sum(item * item for item in self.values))
        self.events.append(f"{self.label}:{value:.2f}")
        return value


class _UpdateParameter(_UpdateTensor):
    """A trainable weight with a gradient tensor."""

    requires_grad = True

    def __init__(self, gradient: float, events: list[str]) -> None:
        super().__init__(0.0, label="weight", events=events)
        self.grad = _UpdateTensor(gradient, label="grad", events=events)


class _UpdateOptimizer:
    """Moves each weight by a fixed fraction of its gradient."""

    def __init__(
        self, parameters: list[_UpdateParameter], events: list[str]
    ) -> None:
        self.parameters = parameters
        self.events = events
        self.seen: list[float] = []

    def step(self) -> None:
        self.events.append("step")
        self.seen = [parameter.grad.values[0] for parameter in self.parameters]
        for parameter in self.parameters:
            parameter.values = tuple(
                weight - 0.5 * gradient
                for weight, gradient in zip(parameter.values, parameter.grad.values)
            )

    def zero_grad(self) -> None:
        pass


def _update_stub(gradients: tuple[float, ...], groups: int):
    """A QwenGenerator-shaped object the real update method can bind to."""
    events: list[str] = []
    parameters = [_UpdateParameter(gradient, events) for gradient in gradients]
    optimizer = _UpdateOptimizer(parameters, events)
    generator = SimpleNamespace(
        trainable=True,
        model=SimpleNamespace(parameters=lambda: iter(parameters)),
        optimizer=optimizer,
        step=0,
        last_grad_norm=None,
        last_update_norm=None,
        _accumulated_groups=groups,
        _pending_groups=[],
    )
    return generator, optimizer, events


def test_update_norm_reports_the_parameter_delta() -> None:
    before = [_UpdateTensor(0.0), _UpdateTensor(0.0)]
    after = [_UpdateTensor(3.0), _UpdateTensor(4.0)]
    assert update_norm(zip(after, before)) == pytest.approx(5.0)


def test_update_norm_is_null_when_the_delta_is_not_finite() -> None:
    before = [_UpdateTensor(0.0)]
    after = [_UpdateTensor(math.inf)]
    assert update_norm(zip(after, before)) is None


def test_apply_update_measures_the_divided_gradient_before_the_step() -> None:
    # Two groups of gradients 3 and 4. The mean over the groups is 1.5 and 2.0,
    # whose norm is 2.5. The raw norm before the divide would be 5.0. The norm
    # calls are recorded in order, so a norm taken after the step is visible.
    generator, optimizer, events = _update_stub((3.0, 4.0), groups=2)

    norm = QwenGenerator.apply_update(generator)

    # The return value catches a norm taken before the divide, which is 5.0.
    assert norm == pytest.approx(2.5)
    assert generator.last_grad_norm == pytest.approx(2.5)
    # The gradient norm is read before the step, which catches a later norm.
    assert events.index("grad:1.50") < events.index("step")
    assert events.index("grad:2.00") < events.index("step")
    # The optimizer applies the mean gradient, not the accumulated sum.
    assert optimizer.seen == pytest.approx([1.5, 2.0])


def test_apply_update_measures_the_weight_change_after_the_step() -> None:
    # The stub moves each weight by half its mean gradient. With two groups,
    # the deltas are 0.75 and 1.0, whose norm is 1.25. Without the divide the
    # deltas would be 1.5 and 2.0, whose norm is 2.5.
    generator, _, events = _update_stub((3.0, 4.0), groups=2)

    QwenGenerator.apply_update(generator)

    assert generator.last_update_norm == pytest.approx(1.25)
    assert events.index("update:0.75") > events.index("step")
    assert events.index("update:1.00") > events.index("step")


def test_apply_update_writes_null_norms_for_an_infinite_gradient() -> None:
    generator, _, _ = _update_stub((math.inf,), groups=1)

    norm = QwenGenerator.apply_update(generator)

    assert norm is None
    assert generator.last_grad_norm is None
    assert generator.last_update_norm is None


def test_group_advantages_are_mean_zero_unit_std() -> None:
    advantages = group_advantages([0.0, 0.5, 1.0])
    assert sum(advantages) == pytest.approx(0.0)
    assert sum(a * a for a in advantages) / len(advantages) == pytest.approx(1.0)


def test_group_advantages_are_zero_when_all_rewards_match() -> None:
    assert group_advantages([0.4, 0.4, 0.4, 0.4]) == [0.0, 0.0, 0.0, 0.0]


def test_group_advantages_rank_rewards() -> None:
    advantages = group_advantages([0.0, 1.0])
    assert advantages[0] < advantages[1]


def test_group_advantages_rejects_an_empty_group() -> None:
    with pytest.raises(ValueError, match="at least one"):
        group_advantages([])


def test_clipped_surrogate_bounds_a_large_positive_advantage() -> None:
    # ratio far above the clip window with a positive advantage: the clipped
    # term (1 + eps) * A wins over the unclipped ratio * A.
    result = clipped_surrogate(ratio=10.0, advantage=1.0, epsilon=0.2)
    assert result == pytest.approx(1.2)


def test_clipped_surrogate_bounds_a_large_negative_advantage() -> None:
    result = clipped_surrogate(ratio=0.01, advantage=-1.0, epsilon=0.2)
    assert result == pytest.approx(-0.8)


def test_clipped_surrogate_leaves_a_unit_ratio_unchanged() -> None:
    assert clipped_surrogate(1.0, 0.3, 0.2) == pytest.approx(0.3)


def test_qwen_module_imports_without_importing_torch() -> None:
    # The real trainer imports torch lazily so the CPU suite can import the
    # module. This test fails if a top-level torch import is ever added.
    import os
    from pathlib import Path
    import subprocess
    import sys

    source_root = Path(__file__).resolve().parents[2] / "src"
    subprocess.run(
        [sys.executable, "-c", "import sys\nimport rlvr.engine.qwen\nassert 'torch' not in sys.modules"],
        env={**os.environ, "PYTHONPATH": str(source_root)},
        check=True,
    )


def test_completion_logprobs_skips_the_prompt_positions() -> None:
    # Vocab 3, prompt length 2, completion [2, 0]. The first completion token is
    # predicted by logits[1], the second by logits[2], never by logits[0].
    logits = [
        [0.0, 0.0, 0.0],  # position 0, a prompt token
        [1.0, 0.0, 0.0],  # predicts the first completion token
        [0.0, 0.0, 2.0],  # predicts the second completion token
        [9.0, 9.0, 9.0],  # unused
    ]
    token_ids = [2, 0]
    result = completion_logprobs(logits, token_ids, prompt_length=2)
    expected = [log_softmax([1.0, 0.0, 0.0])[2], log_softmax([0.0, 0.0, 2.0])[0]]
    assert result == pytest.approx(expected)


def test_ratio_is_one_when_the_policy_has_not_changed() -> None:
    # The property that guards the temperature-mix bug: old and new logprobs are
    # computed by the same function from the same forward logits, so the per-token
    # ratio is exactly 1.0 before any update.
    logits = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    token_ids = [1, 2]
    prompt_length = 1
    old = completion_logprobs(logits, token_ids, prompt_length)
    new = completion_logprobs(logits, token_ids, prompt_length)
    assert old == new
    for old_lp, new_lp in zip(old, new):
        assert math.exp(new_lp - old_lp) == pytest.approx(1.0)


def test_k3_divergence_is_zero_when_distributions_match() -> None:
    assert k3_divergence([0.0, -1.0], [0.0, -1.0]) == pytest.approx(0.0)


def test_k3_divergence_matches_a_known_value() -> None:
    # One token with d = ln(2): exp(d) - d - 1 = 2 - ln(2) - 1.
    expected = 2 - math.log(2) - 1
    assert k3_divergence([math.log(2)], [0.0]) == pytest.approx(expected)


def test_k3_divergence_is_per_token_not_the_summed_ratio() -> None:
    # Two tokens each with d = 1. The per-token estimator is 2*(e - 2), which is
    # not the k3 of the summed log-ratio exp(2) - 2 - 1.
    per_token = k3_divergence([1.0, 1.0], [0.0, 0.0])
    assert per_token == pytest.approx(2 * (math.e - 2))
    summed = math.exp(2.0) - 2.0 - 1.0
    assert per_token != pytest.approx(summed)


def test_truncate_prompt_caps_a_long_prompt() -> None:
    ids = list(range(600))
    kept, truncated = truncate_prompt(ids, 512)
    assert len(kept) == 512
    assert kept == ids[:512]
    assert truncated is True


def test_truncate_prompt_leaves_a_short_prompt() -> None:
    ids = list(range(100))
    kept, truncated = truncate_prompt(ids, 512)
    assert kept == ids
    assert truncated is False


def test_completion_truncated_flags_a_missing_end_token() -> None:
    assert completion_truncated([1, 2, 3], eos_token_id=99) is True
    assert completion_truncated([1, 2, 99], eos_token_id=99) is False


def test_pad_prompts_left_pads_to_the_longest_prompt() -> None:
    padded, mask = pad_prompts([[7, 8, 9], [4]], pad_token_id=0)
    assert padded == [[7, 8, 9], [0, 0, 4]]
    assert mask == [[1, 1, 1], [0, 0, 1]]


def test_pad_prompts_leaves_one_prompt_unpadded() -> None:
    padded, mask = pad_prompts([[5, 6]], pad_token_id=0)
    assert padded == [[5, 6]]
    assert mask == [[1, 1]]


def test_pad_prompts_rejects_an_empty_batch_or_an_empty_prompt() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        pad_prompts([], pad_token_id=0)
    with pytest.raises(ValueError, match="holds no tokens"):
        pad_prompts([[1], []], pad_token_id=0)


def test_check_attention_mask_accepts_a_left_padded_batch() -> None:
    padded, mask = pad_prompts([[7, 8, 9], [4]], pad_token_id=0)
    check_attention_mask(padded, mask, pad_token_id=0)


def test_check_attention_mask_rejects_a_masked_real_token() -> None:
    # A zero in the mask must sit on a pad, so the model would skip a real one.
    with pytest.raises(ValueError, match="masks a real token"):
        check_attention_mask([[0, 9]], [[0, 0]], pad_token_id=0)


def test_check_attention_mask_rejects_padding_on_the_right() -> None:
    with pytest.raises(ValueError, match="left padded"):
        check_attention_mask([[9, 0]], [[1, 0]], pad_token_id=0)


def test_check_attention_mask_rejects_an_all_pad_row() -> None:
    with pytest.raises(ValueError, match="every token"):
        check_attention_mask([[0, 0]], [[0, 0]], pad_token_id=0)


def test_check_attention_mask_rejects_a_bad_flag_and_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="mask 2"):
        check_attention_mask([[9]], [[2]], pad_token_id=0)
    with pytest.raises(ValueError, match="mask rows"):
        check_attention_mask([[9]], [], pad_token_id=0)
    with pytest.raises(ValueError, match="mask entries"):
        check_attention_mask([[9]], [[1, 1]], pad_token_id=0)


def test_seed_rngs_seeds_the_python_rng() -> None:
    import random

    seed_rngs(1234)
    first = random.random()
    seed_rngs(1234)
    second = random.random()
    assert first == second


def test_mean_token_loss_weights_tokens_not_sequences() -> None:
    # A short completion (2 tokens, mean 1.0) and a long one (5 tokens, mean
    # 2.0). Per-token weighting gives 12/7, not the 1.5 sequence mean.
    short = [1.0, 1.0]
    long = [2.0, 2.0, 2.0, 2.0, 2.0]
    assert mean_token_loss([short, long]) == pytest.approx(12.0 / 7.0)


def test_mean_token_loss_equal_means_gives_that_mean() -> None:
    # Both completions have the same mean per-token loss, so the group mean is
    # that value regardless of the length difference.
    short = [1.0, 1.0]
    long = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert mean_token_loss([short, long]) == pytest.approx(1.0)


def test_mean_token_loss_is_zero_for_an_empty_group() -> None:
    assert mean_token_loss([]) == 0.0


def test_profiles_use_the_architecture_output_projection() -> None:
    from rlvr.engine.model_profiles import LFM2, QWEN3, model_profile

    assert model_profile("Qwen/Qwen3-0.6B") == QWEN3
    assert model_profile("LiquidAI/LFM2.5-2.6B") == LFM2
    assert model_profile("/local/model", model_type="lfm2") == LFM2
    assert LFM2.attention_targets == (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"
    )
    with pytest.raises(ValueError, match="unsupported model architecture"):
        model_profile("/local/model", model_type="other")


def test_lfm_requires_its_transformers_5_tokenizer_runtime() -> None:
    from rlvr.engine.model_profiles import LFM2, QWEN3, validate_transformers_version

    validate_transformers_version(QWEN3, "4.57.1")
    validate_transformers_version(LFM2, "5.0.0")
    with pytest.raises(ValueError, match="pip install -e.*vllm"):
        validate_transformers_version(LFM2, "4.57.1")


def test_prompt_template_owns_tokenization_and_lfm_always_thinks() -> None:
    from rlvr.experiment import Prompt
    from rlvr.engine.model_profiles import LFM2, QWEN3
    from rlvr.engine.qwen import QwenGenerator

    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return [1, 2, 3]

    generator = object.__new__(QwenGenerator)
    generator.tokenizer = Tokenizer()
    generator.profile = LFM2
    generator.max_prompt_tokens = 10
    prompt = Prompt(prompt_id="p", problem="Write a function")
    assert generator.encode_prompt(prompt, thinking=False) == ([1, 2, 3], False)
    assert calls[-1] == {"tokenize": True, "add_generation_prompt": True}
    generator.profile = QWEN3
    generator.encode_prompt(prompt, thinking=False)
    assert calls[-1]["enable_thinking"] is False


def test_reasoning_split_uses_the_loaded_tokenizer_marker() -> None:
    from rlvr.engine.qwen import QwenGenerator

    generator = object.__new__(QwenGenerator)
    generator.think_end_token_id = 7
    assert generator._split_at_think_end([1, 7, 151668, 3]) == 2


def test_checkpoint_model_and_adapter_identity_is_checked() -> None:
    from rlvr.engine.model_profiles import LFM2
    from rlvr.engine.qwen import QwenGenerator

    generator = object.__new__(QwenGenerator)
    generator.model_name = "LiquidAI/LFM2.5-2.6B"
    generator.revision = "revision"
    generator.profile = LFM2
    generator.lora_rank = 32
    generator.lora_alpha = 64
    generator.lora_target_modules = LFM2.attention_targets
    adapter = {
        "base_model_name_or_path": generator.model_name,
        "r": 32,
        "lora_alpha": 64,
        "target_modules": list(LFM2.attention_targets),
    }
    generator._validate_checkpoint_identity(adapter, {})  # legacy optimizer state
    with pytest.raises(ValueError, match="base model"):
        generator._validate_checkpoint_identity(
            dict(adapter, base_model_name_or_path="Qwen/Qwen3-0.6B"), {}
        )
    with pytest.raises(ValueError, match="model_revision"):
        generator._validate_checkpoint_identity(adapter, {"model_revision": "wrong"})
    with pytest.raises(ValueError, match="target modules"):
        generator._validate_checkpoint_identity(dict(adapter, target_modules=["o_proj"]), {})


def test_checkpoint_guard_rejects_a_mismatched_lora_placement() -> None:
    from rlvr.engine.model_profiles import LFM2
    from rlvr.engine.qwen import QwenGenerator

    attention = list(LFM2.attention_targets)
    all_linear = attention + [
        "model.layers.0.conv.in_proj",
        "model.layers.0.conv.out_proj",
        "model.layers.0.feed_forward.w1",
        "model.layers.0.feed_forward.w2",
        "model.layers.0.feed_forward.w3",
    ]
    generator = object.__new__(QwenGenerator)
    generator.model_name = "LiquidAI/LFM2.5-2.6B"
    generator.revision = "revision"
    generator.profile = LFM2
    generator.lora_rank = 16
    generator.lora_alpha = 32

    generator.lora_target_modules = tuple(attention)
    attention_adapter = {
        "base_model_name_or_path": generator.model_name,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": all_linear,
    }
    with pytest.raises(ValueError, match="target modules"):
        generator._validate_checkpoint_identity(attention_adapter, {})

    generator.lora_target_modules = tuple(all_linear)
    all_linear_adapter = dict(attention_adapter, target_modules=attention)
    with pytest.raises(ValueError, match="target modules"):
        generator._validate_checkpoint_identity(all_linear_adapter, {})
    generator._validate_checkpoint_identity(
        dict(attention_adapter, target_modules=all_linear), {}
    )


def test_shipped_lfm_profiles_use_the_pinned_model() -> None:
    from pathlib import Path
    from rlvr.core.config import load_config

    root = Path(__file__).resolve().parents[2]
    for name in ("lfm.toml", "lfm-smoke.toml"):
        config = load_config(root / "configs" / name)
        assert config.model_name == "LiquidAI/LFM2.5-2.6B"
        assert len(config.model_revision) == 40
        assert "engine" not in config.raw.get("training", {})


def test_all_linear_targets_adapt_the_lfm_convolution_and_mlp(tmp_path) -> None:
    """The all-linear setting reaches PEFT, on the tiny fake model, no download."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("LFM2.5 requires the optional Transformers 5 environment")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, Lfm2Config, PreTrainedTokenizerFast
    from rlvr.engine.qwen import CausalLMGenerator

    vocabulary = {
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "<bos>": 3,
        "<think>": 4, "</think>": 5, "write": 6, "code": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
        unk_token="<unk>", bos_token="<bos>",
        additional_special_tokens=["<think>", "</think>"],
        chat_template="{{ bos_token }}{{ messages[0]['content'] }}<think>",
    )
    folder = tmp_path / "tiny-lfm-all-linear"
    tokenizer.save_pretrained(folder)
    config = Lfm2Config(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=1, bos_token_id=3,
        full_attn_idxs=[1], block_multiple_of=1, block_auto_adjust_ff_dim=False,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(folder)
    trainer = CausalLMGenerator(
        model_name=str(folder), revision="main", lora_rank=2, lora_alpha=4,
        lora_targets="all-linear", learning_rate=1e-3,
        max_prompt_tokens=16, max_completion_tokens=4,
        clip_epsilon=0.2, kl_beta=0.0, seed=0,
    )

    names = [name for name, _ in trainer.model.named_parameters() if "lora_" in name]
    assert any(".conv." in name for name in names)
    assert any(".feed_forward." in name for name in names)
    assert any(".self_attn." in name for name in names)
    assert any(".conv." in name for name in trainer.lora_target_modules)
    assert any(".feed_forward." in name for name in trainer.lora_target_modules)


@pytest.mark.parametrize("family", ["lfm2", "qwen3"])
def test_cached_logprobs_match_masked_rollout_batch(monkeypatch, family) -> None:
    """Single-sequence convolution forwards must omit masked rollout padding."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    if family == "lfm2" and int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("LFM2.5 requires the optional Transformers 5 environment")
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM
    from rlvr.experiment import Prompt
    from rlvr.engine.model_profiles import LFM2, QWEN3
    from rlvr.engine.qwen import CausalLMGenerator

    torch.manual_seed(17)
    model_options = dict(
        vocab_size=12, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, initializer_range=0.5,
        pad_token_id=0, eos_token_id=1, bos_token_id=3,
    )
    if family == "lfm2":
        from transformers import Lfm2Config
        profile = LFM2
        config = Lfm2Config(
            **model_options, full_attn_idxs=[1],
            block_multiple_of=1, block_auto_adjust_ff_dim=False,
        )
    else:
        from transformers import Qwen3Config
        profile = QWEN3
        config = Qwen3Config(**model_options, head_dim=8)
    model = peft.get_peft_model(
        AutoModelForCausalLM.from_config(config),
        peft.LoraConfig(
            r=2, lora_alpha=4, target_modules=list(profile.attention_targets),
            task_type=peft.TaskType.CAUSAL_LM, lora_dropout=0.0,
        ),
    )
    with torch.no_grad():
        model.get_input_embeddings().weight[0].fill_(0.5)
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(std=0.1)
    trainer = object.__new__(CausalLMGenerator)
    trainer._torch = torch
    trainer._device = torch.device("cpu")
    trainer.model = model
    trainer.profile = profile
    trainer.think_end_token_id = 5
    trainer.kl_beta = 0.1
    trainer.clip_epsilon = 0.2
    trainer.tokenizer = SimpleNamespace(eos_token_id=1, decode=lambda *args, **kwargs: "code")
    prompt_ids = [[0, 0, 3, 6, 7, 4], [3, 6, 7, 6, 7, 4]]
    prompt_masks = [[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    completion = [6, 5, 7, 1]
    batch = torch.tensor([row + completion for row in prompt_ids])
    masks = torch.tensor([row + [1] * len(completion) for row in prompt_masks])
    with torch.no_grad():
        logits = model(batch, attention_mask=masks, use_cache=False).logits[0]
        expected_policy = trainer._completion_logprobs_from_logits(logits, 6, completion)
        with model.disable_adapter():
            logits = model(batch, attention_mask=masks, use_cache=False).logits[0]
        expected_reference = trainer._completion_logprobs_from_logits(logits, 6, completion)

    _, entries = trainer._decode_group(
        Prompt(prompt_id="short", problem="write code"), prompt_ids[0], prompt_masks[0], 6,
        prompt_truncated=False, completion_rows=[completion], caching=True,
    )
    entry = entries[0]
    torch.testing.assert_close(torch.tensor(entry.old_logprobs), expected_policy)
    torch.testing.assert_close(torch.tensor(entry.reference_logprobs), expected_reference)
    assert not torch.allclose(expected_policy, expected_reference)
    if family == "lfm2":
        assert entry.input_ids == [3, 6, 7, 4] + completion
        assert entry.prompt_length == 4
        assert entry.attention_mask == [1] * len(entry.input_ids)
    else:
        assert entry.input_ids == prompt_ids[0] + completion
        assert entry.prompt_length == 6
        assert entry.attention_mask == masks[0].tolist()

    observed = []
    original = trainer._completion_logprobs_from_logits

    def record_new_logprobs(logits, prompt_length, token_ids):
        result = original(logits, prompt_length, token_ids)
        observed.append(result.detach())
        return result

    monkeypatch.setattr(trainer, "_completion_logprobs_from_logits", record_new_logprobs)
    surrogate, kl, _, _ = trainer._group_terms(entry, torch.tensor(1.0), True)
    torch.testing.assert_close(observed[-1], expected_policy)
    (-surrogate + trainer.kl_beta * kl).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)


@pytest.mark.parametrize("family", ["lfm2", "qwen3"])
def test_tiny_model_update_and_resume(tmp_path, family) -> None:
    """Exercise the actual optional runtime without downloading model weights."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if family == "lfm2" and int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("LFM2.5 requires the optional Transformers 5 environment")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from rlvr.experiment import Prompt
    from rlvr.engine.qwen import CausalLMGenerator

    vocabulary = {
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "<bos>": 3,
        "<think>": 4, "</think>": 5, "write": 6, "code": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
        unk_token="<unk>", bos_token="<bos>",
        additional_special_tokens=["<think>", "</think>"],
        chat_template="{{ bos_token }}{{ messages[0]['content'] }}<think>",
    )
    folder = tmp_path / f"tiny-{family}"
    tokenizer.save_pretrained(folder)
    model_options = dict(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, bos_token_id=3,
    )
    if family == "lfm2":
        from transformers import Lfm2Config
        config = Lfm2Config(
            **model_options, full_attn_idxs=[1],
            block_multiple_of=1, block_auto_adjust_ff_dim=False,
        )
    else:
        from transformers import Qwen3Config
        config = Qwen3Config(**model_options, head_dim=8)
    AutoModelForCausalLM.from_config(config).save_pretrained(folder)
    trainer = CausalLMGenerator(
        model_name=str(folder), revision="main", lora_rank=2, lora_alpha=4,
        learning_rate=1e-3, max_prompt_tokens=16, max_completion_tokens=4,
        clip_epsilon=0.2, kl_beta=0.1, seed=0,
    )
    prompt = Prompt(prompt_id="tiny", problem="write code")
    assert trainer.think_end_token_id == 5
    assert trainer.profile.model_type == family
    if family == "lfm2":
        lora_parameters = [
            name for name, _ in trainer.model.named_parameters() if "lora_" in name
        ]
        assert len(lora_parameters) == 8  # A and B for four attention projections
        assert all(".self_attn." in name for name in lora_parameters)
        assert not any(".conv." in name for name in lora_parameters)
    # Emulate a failure halfway through the first group's backward pass.
    parameter = next(p for p in trainer.model.parameters() if p.requires_grad)
    parameter.grad = torch.ones_like(parameter)
    assert trainer._accumulated_groups == 0
    prompt_ids, prompt_truncated = trainer.encode_prompt(prompt)
    completions = [[6, 5, 7, 1], [6, 4, 7, 1]]
    group = trainer.build_generation_batch(
        [prompt], [prompt_ids], [prompt_truncated], [completions], cache=True
    )[0]
    assert parameter.grad is None
    pending = trainer._pending_groups
    trainer.build_generation_batch(
        [prompt], [prompt_ids], [prompt_truncated], [completions[:1]], cache=False
    )
    assert trainer._pending_groups is pending
    before = [p.detach().clone() for p in trainer.model.parameters() if p.requires_grad]
    trainer.update(prompt, group, [0.0, 1.0], measure_entropy=True)
    assert trainer.step == 1
    after = [p.detach().clone() for p in trainer.model.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    checkpoint = tmp_path / "checkpoint"
    trainer.save_checkpoint(checkpoint)
    base_identity = id(trainer.base_model)
    trainer.load_checkpoint(checkpoint)
    assert id(trainer.base_model) == base_identity
    assert trainer.step == 1
    # A resume takes the configured rate, not the one saved with the optimizer.
    trainer.learning_rate = 5e-5
    trainer.load_checkpoint(checkpoint)
    assert all(group["lr"] == 5e-5 for group in trainer.optimizer.param_groups)
    trainer.learning_rate = 1e-3
    trainer.load_checkpoint(checkpoint)
    assert all(torch.equal(a, b) for a, b in zip(after, [
        p.detach() for p in trainer.model.parameters() if p.requires_grad
    ]))
    group = trainer.build_generation_batch(
        [prompt], [prompt_ids], [prompt_truncated], [completions], cache=True
    )[0]
    trainer.update(prompt, group, [1.0, 0.0])
    assert trainer.step == 2

    from safetensors import SafetensorError
    from safetensors.torch import load_file, save_file

    adapter_file = checkpoint / "adapter_model.safetensors"
    saved_weights = load_file(adapter_file)
    current_weights = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }
    for corruption in ("missing", "unexpected", "shape", "truncated"):
        # Also change valid tensors, so validation must precede any model load.
        damaged = {name: value + 1 for name, value in saved_weights.items()}
        key = next(iter(damaged))
        if corruption == "missing":
            damaged.pop(key)
        elif corruption == "unexpected":
            damaged["unexpected_adapter_tensor"] = torch.zeros(1)
        elif corruption == "shape":
            damaged[key] = damaged[key][:1]
        if corruption == "truncated":
            adapter_file.write_bytes(b"invalid safetensors")
        else:
            save_file(damaged, adapter_file)
        with pytest.raises((ValueError, SafetensorError)):
            trainer.load_checkpoint(checkpoint)
        assert trainer.step == 2
        assert all(
            torch.equal(current_weights[name], value)
            for name, value in trainer.model.state_dict().items()
        )
    save_file(saved_weights, adapter_file)

    shorter = Prompt(prompt_id="shorter", problem="write")
    shorter_ids, shorter_truncated = trainer.encode_prompt(shorter)
    groups = trainer.build_generation_batch(
        [prompt, shorter],
        [prompt_ids, shorter_ids],
        [prompt_truncated, shorter_truncated],
        [[completions[0]], [completions[0]]],
        cache=False,
    )
    assert len(groups) == 2
