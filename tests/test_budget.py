"""Tests for the thinking budget schedule."""

from __future__ import annotations

import pytest

from rlvr.budget import DEFAULT_THINKING_BUDGET_START, thinking_budget_for_stage


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

