"""Thinking budget schedule for the first curriculum stages.

The model learns to close its reasoning block sooner if the budget is short
enough that closing is reachable. The budget starts at a configured value and
doubles at each curriculum stage, up to the completion cap. A rollout that
reaches the budget without closing is forced to end, so it scores as
truncated and the group keeps some variance.

Pure Python, no torch. The generation loop calls `row_should_force_end` for
each row; the tensor work stays in `qwen.py`.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_THINKING_BUDGET_START",
    "budget_stop_is_truncation",
    "row_should_force_end",
    "thinking_budget_for_stage",
]

#: Tokens allowed for the reasoning block at the first stage.
DEFAULT_THINKING_BUDGET_START = 768


def thinking_budget_for_stage(start: int, stage_index: int, cap: int) -> int:
    """The reasoning budget at one curriculum stage, never above `cap`.

    Stage 0 gets `start`, and each later stage doubles. The cap is the
    completion cap, so the budget can never exceed the room generation has.
    """
    if isinstance(start, bool) or not isinstance(start, int) or start < 1:
        raise ValueError(f"start must be a positive integer, got {start!r}")
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
        raise ValueError(
            f"stage_index must be a non-negative integer, got {stage_index!r}"
        )
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"cap must be a positive integer, got {cap!r}")
    return min(cap, start * 2**stage_index)


def budget_stop_is_truncation(
    generated_len: int,
    budget: int,
    has_marker: bool,
    already_truncated: bool = False,
) -> bool:
    """Whether a completion is scored as truncated because of the budget.

    A natural stop at the completion cap is already truncated. Otherwise a
    completion that reached the thinking budget without closing is a budget
    stop, and it takes the truncation penalty.
    """
    if already_truncated:
        return True
    return row_should_force_end(generated_len, budget, has_marker)


def row_should_force_end(generated_len: int, budget: int, has_marker: bool) -> bool:
    """Whether one row must be ended because it ran out of thinking budget.

    A row that already closed its block is never forced. A row that reaches the
    budget without closing is ended, so it cannot spend the rest of the
    completion cap on an unclosed trace.
    """
    if isinstance(generated_len, bool) or not isinstance(generated_len, int):
        raise TypeError(f"generated_len must be an integer, got {generated_len!r}")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")
    return generated_len >= budget and not has_marker
