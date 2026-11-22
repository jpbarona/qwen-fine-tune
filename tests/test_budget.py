"""Tests for the fixed thinking budget."""

from __future__ import annotations

from rlvr.budget import DEFAULT_THINKING_BUDGET


def test_the_default_budget_is_one_fixed_value() -> None:
    assert DEFAULT_THINKING_BUDGET == 1024
