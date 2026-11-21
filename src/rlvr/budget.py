"""Thinking budget schedule for the first curriculum stages.

The budget starts at a configured value and doubles at each curriculum stage,
up to the completion cap. It is a config value only. vLLM 0.34 has no
per-request logits processors, so the sampler does not enforce the budget and
a rollout is not ended when it reaches it.

Pure Python, no torch.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_THINKING_BUDGET_START",
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
