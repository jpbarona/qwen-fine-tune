"""Tests for the pure helpers in qwen.py. No torch is imported."""

from __future__ import annotations

import math

import pytest

from rlvr.qwen import (
    ATTENTION_TARGET_MODULES,
    categorical_entropy,
    clipped_surrogate,
    extract_code_from_answer,
    group_advantages,
    split_thinking_and_code,
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
