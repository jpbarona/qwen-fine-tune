"""Tests for the thinking budget schedule."""

from __future__ import annotations

import pytest

from rlvr.budget import (
    DEFAULT_THINKING_BUDGET_START,
    row_should_force_end,
    thinking_budget_for_stage,
)


def test_the_budget_starts_at_the_default_and_doubles_per_stage() -> None:
    assert DEFAULT_THINKING_BUDGET_START == 768
    assert [
        thinking_budget_for_stage(DEFAULT_THINKING_BUDGET_START, stage, 2048)
        for stage in range(4)
    ] == [768, 1536, 2048, 2048]


def test_the_budget_never_exceeds_the_completion_cap() -> None:
    assert thinking_budget_for_stage(1000, 0, 800) == 800
    assert thinking_budget_for_stage(768, 5, 2048) == 2048


def test_the_budget_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        thinking_budget_for_stage(0, 0, 2048)
    with pytest.raises(ValueError):
        thinking_budget_for_stage(768, -1, 2048)
    with pytest.raises(ValueError):
        thinking_budget_for_stage(768, 0, 0)


def test_a_row_is_forced_to_end_only_when_the_budget_is_spent_unclosed() -> None:
    assert row_should_force_end(10, 768, has_marker=False) is False
    assert row_should_force_end(767, 768, has_marker=False) is False
    assert row_should_force_end(768, 768, has_marker=False) is True
    assert row_should_force_end(900, 768, has_marker=False) is True


def test_a_closed_row_is_never_forced() -> None:
    assert row_should_force_end(768, 768, has_marker=True) is False
    assert row_should_force_end(2000, 768, has_marker=True) is False


def test_the_forced_end_check_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        row_should_force_end(10, 0, has_marker=False)
    with pytest.raises(TypeError):
        row_should_force_end(1.5, 768, has_marker=False)
