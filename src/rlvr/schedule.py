"""The step schedule for the training pool.

The step draw asks a schedule for the prompts it should run. The schedule
returns a plan with two lists: the train prompts that carry the step and the
parked prompts that join it for one re-screen. Parked rollouts use the same
generator call and the same optimizer update as the train rollouts, so their
groups can still produce a gradient.

Two rules move a prompt between the pools after a step:

    promote   a parked group with at least `promotion_bar` passing rollouts
    demote    a train prompt after `demote_after` groups with no passing rollout

The rotation is deterministic. The parked order comes from the run seed and the
slice position comes from the step number, so a resume at step N plans the same
slice as the original run at step N. No process state, clock or hash order is
an input.

`Spread` is the first policy. It takes a fixed slice of parked prompts every
step. The slice is the smaller of the cap and the count that visits the whole
parked pool once every `refresh_steps` steps. Train prompts fill the rest of
the step.

`Sweep` is the second policy. It covers the whole parked pool in `m` consecutive
all-parked steps, then trains for `gap_ratio * m` steps with no parked prompts.
A parked group is labelled in the step it runs, so a resume loses no sweep
outcome. The cycle position is saved beside the label state, so a resume
continues the same sweep at the same index.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from .pools import DEFAULT_PROMOTION_BAR

if TYPE_CHECKING:  # pragma: no cover - import used only by type checkers
    from .experiment import Prompt

__all__ = [
    "DEFAULT_DEMOTE_AFTER",
    "DEFAULT_PROMOTION_BAR",
    "DEFAULT_REFRESH_STEPS",
    "DEFAULT_SWEEP_GAP_RATIO",
    "MAX_SWEEP_GAP_RATIO",
    "MIN_SWEEP_GAP_RATIO",
    "GroupOutcome",
    "LabelUpdate",
    "Schedule",
    "Spread",
    "StepPlan",
    "Sweep",
    "label_from_row",
    "streak_from_row",
    "sweep_order_from_row",
    "sweep_position_from_row",
]

#: A parked prompt is promoted at two passing rollouts or more. The value is
#: the pool module's promotion bar, so the screen and the schedule cannot drift.
#: (Imported above and re-exported here.)

#: A train prompt is demoted after ten consecutive groups with no passing
#: rollout.
DEFAULT_DEMOTE_AFTER = 10

#: The parked pool is meant to be visited once per this many steps.
DEFAULT_REFRESH_STEPS = 40

#: The gap between sweeps is this many times the sweep length. Three is the
#: default, so a cycle spends about a quarter of its steps on parked work.
DEFAULT_SWEEP_GAP_RATIO = 3

#: The allowed range for the sweep gap ratio.
MIN_SWEEP_GAP_RATIO = 3
MAX_SWEEP_GAP_RATIO = 10

#: The two labels a prompt can hold in the training schedule.
TRAIN_LABEL = "train"
PARKED_LABEL = "parked"


@dataclass(frozen=True)
class StepPlan:
    """The prompts one step runs.

    `train` holds the prompts that fill the step and `parked` holds the re-screen
    slice. The order is the order the generator receives.
    """

    train: tuple["Prompt", ...] = ()
    parked: tuple["Prompt", ...] = ()

    @property
    def prompts(self) -> tuple["Prompt", ...]:
        """Every prompt in the step, train first."""
        return self.train + self.parked

    def __len__(self) -> int:
        return len(self.train) + len(self.parked)


@dataclass(frozen=True)
class GroupOutcome:
    """What one prompt group produced in a step.

    `passed` is the number of rollouts that passed every task test. `samples`
    is the number of rollouts in the group. `parked` says which pool the group
    came from, because the promotion and demotion rules differ.
    """

    prompt_id: str
    passed: int
    samples: int
    parked: bool

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("a group outcome needs a non-empty prompt id")
        if (
            isinstance(self.samples, bool)
            or not isinstance(self.samples, int)
            or self.samples < 1
        ):
            raise ValueError(
                f"prompt {self.prompt_id!r}: samples must be an integer of at "
                f"least 1, got {self.samples!r}"
            )
        if (
            isinstance(self.passed, bool)
            or not isinstance(self.passed, int)
            or not 0 <= self.passed <= self.samples
        ):
            raise ValueError(
                f"prompt {self.prompt_id!r}: passed must be an integer between "
                f"0 and {self.samples}, got {self.passed!r}"
            )


@dataclass(frozen=True)
class LabelUpdate:
    """One pool change written to the run's label file."""

    step: int
    prompt_id: str
    label: str
    reason: str
    passed: int
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "prompt_id": self.prompt_id,
            "label": self.label,
            "reason": self.reason,
            "passed": self.passed,
            "samples": self.samples,
        }


class Schedule(Protocol):
    """The step-draw seam shared by every pool policy."""

    def plan(
        self,
        step: int,
        train_pool: Sequence["Prompt"],
        parked_pool: Sequence["Prompt"],
    ) -> StepPlan: ...

    def observe(
        self, step: int, outcomes: Sequence[GroupOutcome]
    ) -> tuple[LabelUpdate, ...]: ...


def _prompt_id(prompt: object) -> str:
    """The id of a prompt from any object with the field."""
    value = getattr(prompt, "prompt_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every prompt in a plan needs a non-empty prompt_id")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1, got {value!r}")
    return value


def _observe_outcome(
    step: int,
    outcome: GroupOutcome,
    *,
    promotion_bar: int,
    demote_after: int,
    failure_streak: dict[str, int],
) -> LabelUpdate | None:
    """Apply the pool rules to one group and return its update, if any.

    A parked group promotes at `promotion_bar` passing rollouts. A train group
    with no passing rollout extends its full-failure streak, and `demote_after`
    consecutive groups demote it. A train group with any passing rollout clears
    the streak. A parked group never changes a train streak.
    """
    if outcome.parked:
        if outcome.passed >= promotion_bar:
            failure_streak.pop(outcome.prompt_id, None)
            return LabelUpdate(
                step=step,
                prompt_id=outcome.prompt_id,
                label=TRAIN_LABEL,
                reason="promoted",
                passed=outcome.passed,
                samples=outcome.samples,
            )
        return None

    if outcome.passed == 0:
        streak = failure_streak.get(outcome.prompt_id, 0) + 1
    else:
        streak = 0

    update: LabelUpdate | None = None
    if streak >= demote_after:
        update = LabelUpdate(
            step=step,
            prompt_id=outcome.prompt_id,
            label=PARKED_LABEL,
            reason="demoted",
            passed=outcome.passed,
            samples=outcome.samples,
        )
        streak = 0
    failure_streak[outcome.prompt_id] = streak
    return update


def _saved_streak(failure_streak: dict[str, int], prompt_id: str) -> int:
    """The current full-failure streak for one prompt."""
    return failure_streak.get(prompt_id, 0)


def _restore_saved_streak(
    failure_streak: dict[str, int], prompt_id: str, streak: int
) -> None:
    """Set one prompt's saved streak. Bad rows are ignored."""
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        return
    if isinstance(streak, bool) or not isinstance(streak, int) or streak < 0:
        return
    if streak == 0:
        failure_streak.pop(prompt_id, None)
    else:
        failure_streak[prompt_id] = streak


@dataclass
class Spread:
    """A fixed parked slice every step, in a seeded rotation.

    `prompts_per_step` is the full step size. `cap` is the largest parked
    slice. `refresh_steps` is the target number of steps to visit the whole
    parked pool once. The slice is the smaller of the cap and
    `ceil(parked_count / refresh_steps)`.
    """

    prompts_per_step: int
    cap: int
    refresh_steps: int
    seed: int = 0
    promotion_bar: int = DEFAULT_PROMOTION_BAR
    demote_after: int = DEFAULT_DEMOTE_AFTER
    name: str = field(default="spread", init=False)
    _failure_streak: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _positive_int("prompts_per_step", self.prompts_per_step)
        _positive_int("refresh_steps", self.refresh_steps)
        _positive_int("promotion_bar", self.promotion_bar)
        _positive_int("demote_after", self.demote_after)
        if isinstance(self.cap, bool) or not isinstance(self.cap, int) or self.cap < 0:
            raise ValueError(f"cap must be an integer of at least 0, got {self.cap!r}")
        if self.cap > self.prompts_per_step:
            raise ValueError(
                f"cap must not exceed prompts_per_step, got {self.cap} and "
                f"{self.prompts_per_step}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")

    def parked_count(self, parked: int) -> int:
        """The parked slice for a pool of `parked` prompts."""
        if parked < 0:
            raise ValueError(f"parked count must not be negative, got {parked}")
        if parked == 0:
            return 0
        return min(self.cap, math.ceil(parked / self.refresh_steps))

    def plan(
        self,
        step: int,
        train_pool: Sequence["Prompt"],
        parked_pool: Sequence["Prompt"],
    ) -> StepPlan:
        """The train prompts and parked slice for one step.

        The parked order is a fixed shuffle by `seed`. The window moves by the
        slice size each step, so the slice position is `(step - 1) * k` modulo
        the parked count. That position depends only on the step number, not on
        how many steps this process has run.
        """
        _positive_int("step", step)
        parked = sorted(parked_pool, key=_prompt_id)
        train = sorted(train_pool, key=_prompt_id)

        slice_size = self.parked_count(len(parked))
        train_size = self.prompts_per_step - slice_size

        order = [_prompt_id(prompt) for prompt in parked]
        random.Random(self.seed).shuffle(order)
        by_id = {_prompt_id(prompt): prompt for prompt in parked}

        if slice_size:
            start = ((step - 1) * slice_size) % len(order)
            parked_slice = tuple(
                by_id[order[(start + index) % len(order)]]
                for index in range(slice_size)
            )
        else:
            parked_slice = ()

        if train:
            train_start = ((step - 1) * train_size) % len(train)
            train_block = tuple(
                train[(train_start + index) % len(train)]
                for index in range(train_size)
            )
        else:
            train_block = ()

        return StepPlan(train=train_block, parked=parked_slice)

    def observe(
        self, step: int, outcomes: Sequence[GroupOutcome]
    ) -> tuple[LabelUpdate, ...]:
        """Apply the promotion and demotion rules to one step's groups.

        A parked group promotes at `promotion_bar` passing rollouts. A train
        group with no passing rollout extends that prompt's streak, and ten
        consecutive groups demote it. A train group with any passing rollout
        clears the streak. Parked groups never change a train streak.
        """
        _positive_int("step", step)
        updates: list[LabelUpdate] = []

        for outcome in outcomes:
            update = _observe_outcome(
                step,
                outcome,
                promotion_bar=self.promotion_bar,
                demote_after=self.demote_after,
                failure_streak=self._failure_streak,
            )
            if update is not None:
                updates.append(update)

        return tuple(updates)

    def failure_streak(self, prompt_id: str) -> int:
        """The current full-failure streak for a train prompt."""
        return _saved_streak(self._failure_streak, prompt_id)

    def restore_streak(self, prompt_id: str, streak: int) -> None:
        """Set a prompt's failure streak from a saved label row.

        A resume replays the rows in file order, so a later row with a zero
        streak clears an earlier nonzero one. A negative or non-integer count
        is ignored, because a hand-edited row must not break the run.
        """
        _restore_saved_streak(self._failure_streak, prompt_id, streak)


@dataclass
class Sweep:
    """A whole-pool sweep in `m` steps, then a gap of training steps.

    `prompts_per_step` is the target step size. With `P` parked prompts, the
    sweep length is `m = ceil(P / prompts_per_step)` and the gap is
    `gap_ratio * m` training steps. Every step in the sweep is parked only, so
    no train prompt appears. A final sweep step may hold fewer prompts when the
    parked count does not divide evenly. A sweep step never repeats a prompt,
    because the budget is a target and not a floor. The seeded order is fixed at
    the start of the sweep and saved with the position. A prompt promoted in an
    earlier step is skipped, so it cannot run twice and no prompt is skipped.

    The schedule keeps a cycle position. Position `0` to `m - 1` is a sweep
    step and the rest of the cycle is a training step. The position is saved
    beside the label state after every step, so a resume continues the same
    sweep at the same index even when the parked count changed.
    """

    prompts_per_step: int
    gap_ratio: int = DEFAULT_SWEEP_GAP_RATIO
    seed: int = 0
    promotion_bar: int = DEFAULT_PROMOTION_BAR
    demote_after: int = DEFAULT_DEMOTE_AFTER
    name: str = field(default="sweep", init=False)
    _failure_streak: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _position: int = field(default=0, init=False, repr=False)
    _order: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _saved_order: tuple[str, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _positive_int("prompts_per_step", self.prompts_per_step)
        _positive_int("promotion_bar", self.promotion_bar)
        _positive_int("demote_after", self.demote_after)
        if (
            isinstance(self.gap_ratio, bool)
            or not isinstance(self.gap_ratio, int)
            or not MIN_SWEEP_GAP_RATIO <= self.gap_ratio <= MAX_SWEEP_GAP_RATIO
        ):
            raise ValueError(
                f"gap_ratio must be an integer from {MIN_SWEEP_GAP_RATIO} to "
                f"{MAX_SWEEP_GAP_RATIO}, got {self.gap_ratio!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")

    @property
    def position(self) -> int:
        """The cycle position the next step will use."""
        return self._position

    def sweep_length(self, parked: int) -> int:
        """The number of consecutive steps that cover `parked` prompts."""
        if parked < 0:
            raise ValueError(f"parked count must not be negative, got {parked}")
        if parked == 0:
            return 0
        return math.ceil(parked / self.prompts_per_step)

    def gap_length(self, parked: int) -> int:
        """The number of training steps after a sweep of `parked` prompts."""
        return self.gap_ratio * self.sweep_length(parked)

    def cycle_length(self, parked: int) -> int:
        """The sweep length plus the gap for `parked` prompts."""
        return self.sweep_length(parked) + self.gap_length(parked)

    def _train_block(
        self, step: int, train: Sequence["Prompt"]
    ) -> tuple["Prompt", ...]:
        """A full training step from the train pool, cycling when it is short."""
        if not train:
            return ()
        start = ((step - 1) * self.prompts_per_step) % len(train)
        return tuple(
            train[(start + index) % len(train)]
            for index in range(self.prompts_per_step)
        )

    def plan(
        self,
        step: int,
        train_pool: Sequence["Prompt"],
        parked_pool: Sequence["Prompt"],
    ) -> StepPlan:
        """The prompts one step runs, from the saved cycle position.

        With no parked prompts the sweep never starts, the position stays at
        zero, and every step trains normally. A sweep step plans parked prompts
        only. A gap step plans train prompts only.
        """
        _positive_int("step", step)
        parked = sorted(parked_pool, key=_prompt_id)
        train = sorted(train_pool, key=_prompt_id)

        sweep_steps = self.sweep_length(len(parked))
        if sweep_steps == 0:
            self._position = 0
            self._order = None
            self._saved_order = None
            return StepPlan(train=self._train_block(step, train))

        cycle = sweep_steps + self.gap_ratio * sweep_steps
        self._position %= cycle
        position = self._position

        if position < sweep_steps:
            if position == 0 or self._order is None:
                order = [_prompt_id(prompt) for prompt in parked]
                random.Random(self.seed).shuffle(order)
                self._order = tuple(order)
            by_id = {_prompt_id(prompt): prompt for prompt in parked}
            start = position * self.prompts_per_step
            block = self._order[start : start + self.prompts_per_step]
            parked_slice = tuple(
                by_id[prompt_id] for prompt_id in block if prompt_id in by_id
            )
            next_position = (position + 1) % cycle
            self._position = next_position
            self._saved_order = (
                self._order if next_position < sweep_steps else None
            )
            return StepPlan(parked=parked_slice)

        self._order = None
        self._saved_order = None
        self._position = (position + 1) % cycle
        return StepPlan(train=self._train_block(step, train))

    def observe(
        self, step: int, outcomes: Sequence[GroupOutcome]
    ) -> tuple[LabelUpdate, ...]:
        """Apply the promotion and demotion rules to one step's groups.

        A parked group is decided in the step it runs, so a sweep promotion does
        not wait for the rest of the sweep and a resume loses nothing. A train
        group in a gap step follows the spread demotion rule.
        """
        _positive_int("step", step)
        updates: list[LabelUpdate] = []

        for outcome in outcomes:
            update = _observe_outcome(
                step,
                outcome,
                promotion_bar=self.promotion_bar,
                demote_after=self.demote_after,
                failure_streak=self._failure_streak,
            )
            if update is not None:
                updates.append(update)

        return tuple(updates)

    def state_row(self, step: int) -> dict[str, Any]:
        """The saved cycle position after this step, for the label file.

        A position inside a sweep also saves the shuffled order. A resume then
        reuses the order from the sweep start, so a prompt promoted in an
        earlier step cannot run twice and no prompt is skipped.
        """
        row: dict[str, Any] = {"step": step, "sweep_position": self._position}
        if self._saved_order is not None:
            row["sweep_order"] = list(self._saved_order)
        return row

    def restore_order(self, order: Sequence[str]) -> None:
        """Set the sweep order from a saved label row. Bad values are ignored."""
        if isinstance(order, str) or not isinstance(order, Sequence):
            return
        ids = tuple(order)
        if not ids:
            return
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            return
        self._order = ids

    def restore_position(self, position: int) -> None:
        """Set the cycle position from a saved label row.

        A negative or non-integer value is ignored, because a hand-edited row
        must not break the run.
        """
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            return
        self._position = position

    def failure_streak(self, prompt_id: str) -> int:
        """The current full-failure streak for a train prompt."""
        return _saved_streak(self._failure_streak, prompt_id)

    def restore_streak(self, prompt_id: str, streak: int) -> None:
        """Set a prompt's failure streak from a saved label row."""
        _restore_saved_streak(self._failure_streak, prompt_id, streak)


def label_from_row(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """The prompt id and label in a stored label row, or None when malformed.

    The run store validates its own rows. This helper keeps the experiment's
    resume path small and tolerant of a row from an older writer.
    """
    prompt_id = row.get("prompt_id")
    label = row.get("label")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        return None
    if label not in (TRAIN_LABEL, PARKED_LABEL):
        return None
    return prompt_id, label


def streak_from_row(row: Mapping[str, Any]) -> tuple[str, int] | None:
    """The prompt id and saved failure streak, or None when the row has none."""
    prompt_id = row.get("prompt_id")
    streak = row.get("streak")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        return None
    if isinstance(streak, bool) or not isinstance(streak, int) or streak < 0:
        return None
    return prompt_id, streak


def sweep_position_from_row(row: Mapping[str, Any]) -> int | None:
    """The saved sweep cycle position, or None when the row has none."""
    position = row.get("sweep_position")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        return None
    return position


def sweep_order_from_row(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The saved sweep order, or None when the row has none or it is malformed."""
    order = row.get("sweep_order")
    if not isinstance(order, (list, tuple)) or not order:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in order):
        return None
    return tuple(order)
