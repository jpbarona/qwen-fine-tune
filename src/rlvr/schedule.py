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
    "GroupOutcome",
    "LabelUpdate",
    "Schedule",
    "Spread",
    "StepPlan",
]

#: A parked prompt is promoted at two passing rollouts or more. The value is
#: the pool module's promotion bar, so the screen and the schedule cannot drift.
#: (Imported above and re-exported here.)

#: A train prompt is demoted after ten consecutive groups with no passing
#: rollout.
DEFAULT_DEMOTE_AFTER = 10

#: The parked pool is meant to be visited once per this many steps.
DEFAULT_REFRESH_STEPS = 40

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
            if outcome.parked:
                if outcome.passed >= self.promotion_bar:
                    updates.append(
                        LabelUpdate(
                            step=step,
                            prompt_id=outcome.prompt_id,
                            label=TRAIN_LABEL,
                            reason="promoted",
                            passed=outcome.passed,
                            samples=outcome.samples,
                        )
                    )
                    self._failure_streak.pop(outcome.prompt_id, None)
                continue

            if outcome.passed == 0:
                streak = self._failure_streak.get(outcome.prompt_id, 0) + 1
            else:
                streak = 0

            if streak >= self.demote_after:
                updates.append(
                    LabelUpdate(
                        step=step,
                        prompt_id=outcome.prompt_id,
                        label=PARKED_LABEL,
                        reason="demoted",
                        passed=outcome.passed,
                        samples=outcome.samples,
                    )
                )
                streak = 0
            self._failure_streak[outcome.prompt_id] = streak

        return tuple(updates)

    def failure_streak(self, prompt_id: str) -> int:
        """The current full-failure streak for a train prompt."""
        return self._failure_streak.get(prompt_id, 0)


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
