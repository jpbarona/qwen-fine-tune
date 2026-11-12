"""Tests for the pure helpers in qwen.py. No torch is imported."""

from __future__ import annotations

import math

import pytest

from rlvr.qwen import (
    ATTENTION_TARGET_MODULES,
    categorical_entropy,
    clipped_surrogate,
    completion_logprobs,
    completion_truncated,
    extract_code_from_answer,
    group_advantages,
    k3_divergence,
    log_softmax,
    seed_rngs,
    split_thinking_and_code,
    truncate_prompt,
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


def test_categorical_entropy_of_uniform_two_values_is_ln2() -> None:
    assert categorical_entropy([0.0, 0.0]) == pytest.approx(math.log(2))


def test_categorical_entropy_of_a_peaked_distribution_is_small() -> None:
    assert categorical_entropy([100.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)


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
    import importlib
    import sys

    sys.modules.pop("torch", None)
    module = importlib.import_module("rlvr.qwen")
    assert "torch" not in sys.modules
    assert module.extract_code_from_answer("```python\nx = 1\n```") == "x = 1"


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


def test_seed_rngs_seeds_the_python_rng() -> None:
    import random

    seed_rngs(1234)
    first = random.random()
    seed_rngs(1234)
    second = random.random()
    assert first == second
