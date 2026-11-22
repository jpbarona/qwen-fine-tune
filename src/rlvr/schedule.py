"""The step schedule for the training pool.

The step draw asks a schedule for the prompts it should run. The schedule
returns a plan with two lists: the train prompts that carry the step and the
parked prompts that join it for one re-screen. Parked rollouts use the same
generator call and the same optimizer update as the train rollouts, so their
groups can still produce a gradient.

Two rules move a prompt between the pools after a step:

    promote   a parked group with at least `promotion_bar` passing rollouts
    demote    a train prompt after `demote_after` groups with no passing rollout

A passing train group does not clear the zero-pass count. A demotion resets it.

`Spread` is the first policy. It runs a parked block and a train block every
step. Each pool has a queue of prompt ids, saved after each step. A queue covers
every prompt once before any prompt repeats, and a resume continues the same
order. No process state, clock or hash order is an input.

A prompt that changes pool leaves its queue and joins the back of the other
queue. So the new member waits one tour of its new pool.

`Sweep` is the second policy. It covers the whole parked pool in `m` consecutive
all-parked steps, then trains for `gap_ratio * m` steps with no parked prompts.
The parked steps use a fixed order and a cycle position. The gap steps take
their train blocks from a train queue, so the train pool is covered once before
a prompt repeats. A parked group is labelled in the step it runs, so a resume
loses no sweep outcome. The cycle position, the sweep order and the train queue
are saved beside the label state, so a resume continues the same sweep at the
same index and the same train rotation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from .pools import DEFAULT_PROMOTION_BAR
from .queues import PromptQueue

if TYPE_CHECKING:  # pragma: no cover - import used only by type checkers
    from .experiment import Prompt

__all__ = [
    "DEFAULT_DEMOTE_AFTER",
    "DEFAULT_PROMOTION_BAR",
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
    "sweep_steps_from_row",
]

#: A parked prompt is promoted at two passing rollouts or more. The value is
#: the pool module's promotion bar, so the screen and the schedule cannot drift.
#: (Imported above and re-exported here.)

#: A train prompt is demoted after three groups with no passing rollout. A
#: passing group does not clear the count.
DEFAULT_DEMOTE_AFTER = 3

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
    block. The order is the order the generator receives.
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


def _keep_and_fill(queue: PromptQueue, pool_ids: Sequence[str]) -> None:
    """Drop ids that left the pool and append ids that joined it.

    This runs every step against the live pool. A crash between a label row
    and a queues row leaves the saved queue one move behind. The label
    replay moves the pool, so the next plan must repair the queue from the
    pool. A new id joins at the back, as a live move would put it.
    """
    queue.keep(pool_ids)
    held = set(queue.as_list())
    for prompt_id in sorted(set(pool_ids)):
        if prompt_id not in held:
            queue.append(prompt_id)


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
    with no passing rollout adds one to its count, and a count of
    `demote_after` demotes it on a zero-pass group. A passing train group
    leaves the count alone. A demotion resets the count to zero. A parked group
    never changes a train count.
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

    streak = failure_streak.get(outcome.prompt_id, 0)
    if outcome.passed == 0:
        streak += 1

    update: LabelUpdate | None = None
    if outcome.passed == 0 and streak >= demote_after:
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
    """The current zero-pass count for one prompt."""
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
    """A parked block and a train block every step, in saved queues.

    `prompts_per_step` is the full step size. The parked block is half of it,
    with a minimum of four prompts, and it is never the whole step. So a step of
    19 prompts runs 9 parked and 10 train. A step of one prompt runs no parked
    prompts, because the rule keeps at least one train prompt.

    Each pool has a queue of prompt ids. `plan` takes the parked block from the
    parked queue and the rest of the step from the train queue. A queue covers
    every prompt once before any prompt repeats. A prompt that changes pool
    leaves its queue and joins the back of the other queue. The queues are
    saved after each step and restored on a resume.
    """

    prompts_per_step: int
    seed: int = 0
    promotion_bar: int = DEFAULT_PROMOTION_BAR
    demote_after: int = DEFAULT_DEMOTE_AFTER
    name: str = field(default="spread", init=False)
    _failure_streak: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _parked_queue: PromptQueue = field(
        default_factory=PromptQueue, init=False, repr=False
    )
    _train_queue: PromptQueue = field(
        default_factory=PromptQueue, init=False, repr=False
    )
    _queues_built: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _positive_int("prompts_per_step", self.prompts_per_step)
        _positive_int("promotion_bar", self.promotion_bar)
        _positive_int("demote_after", self.demote_after)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")

    def parked_count(self) -> int:
        """The planned parked block for one step.

        Half the step, with a minimum of four, capped at one less than the step
        so at least one train prompt remains.
        """
        half = max(4, self.prompts_per_step // 2)
        return min(half, self.prompts_per_step - 1)

    def planned_parked_share(self) -> float:
        """The share of one step's prompts that the parked block takes.

        The spread share is a feature of the step size, not of the parked pool
        size.
        """
        return self.parked_count() / self.prompts_per_step

    def _ensure_queues(
        self, train_ids: Sequence[str], parked_ids: Sequence[str]
    ) -> None:
        """Build both queues from the pools on the first step."""
        if self._queues_built:
            return
        self._parked_queue.build(parked_ids, self.seed)
        self._train_queue.build(train_ids, self.seed)
        self._queues_built = True

    def plan(
        self,
        step: int,
        train_pool: Sequence["Prompt"],
        parked_pool: Sequence["Prompt"],
    ) -> StepPlan:
        """The train and parked blocks for one step, from the queues."""
        _positive_int("step", step)
        parked_by_id = {_prompt_id(prompt): prompt for prompt in parked_pool}
        train_by_id = {_prompt_id(prompt): prompt for prompt in train_pool}

        self._ensure_queues(
            list(train_by_id),
            list(parked_by_id),
        )
        _keep_and_fill(self._parked_queue, list(parked_by_id))
        _keep_and_fill(self._train_queue, list(train_by_id))

        parked_block = self._parked_queue.take(self.parked_count())
        train_size = self.prompts_per_step - len(parked_block)
        train_block = self._train_queue.take(train_size)

        return StepPlan(
            train=tuple(train_by_id[prompt_id] for prompt_id in train_block),
            parked=tuple(parked_by_id[prompt_id] for prompt_id in parked_block),
        )

    def observe(
        self, step: int, outcomes: Sequence[GroupOutcome]
    ) -> tuple[LabelUpdate, ...]:
        """Apply the promotion and demotion rules to one step's groups.

        A parked group promotes at `promotion_bar` passing rollouts. A train
        group with no passing rollout adds one to that prompt's count, and
        `demote_after` counts demote it. A passing train group leaves the count
        alone. Parked groups never change a train count.

        A promoted id leaves the parked queue and joins the back of the train
        queue. A demoted id leaves the train queue and joins the back of the
        parked queue.
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
                self._move_in_queues(update)

        return tuple(updates)

    def _move_in_queues(self, update: LabelUpdate) -> None:
        """Move one changed id to the back of its new queue."""
        if not self._queues_built:
            return
        if update.label == TRAIN_LABEL:
            self._parked_queue.remove(update.prompt_id)
            self._train_queue.append(update.prompt_id)
        else:
            self._train_queue.remove(update.prompt_id)
            self._parked_queue.append(update.prompt_id)

    def state_row(self, step: int) -> dict[str, Any]:
        """The saved queues after this step, for the label file."""
        return {
            "step": step,
            "parked_queue": self._parked_queue.as_list(),
            "train_queue": self._train_queue.as_list(),
        }

    def restore_state(
        self, parked_queue: PromptQueue, train_queue: PromptQueue
    ) -> None:
        """Set both queues from a saved `queues` row.

        The queues arrive reconciled with the current pools. The next plan
        reconciles them again, because a label row can follow the queues row and
        move a prompt after the restore.
        """
        if not isinstance(parked_queue, PromptQueue) or not isinstance(
            train_queue, PromptQueue
        ):
            raise TypeError("restore_state needs two PromptQueue objects")
        self._parked_queue = parked_queue
        self._train_queue = train_queue
        self._queues_built = True

    def failure_streak(self, prompt_id: str) -> int:
        """The current zero-pass count for a train prompt."""
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
    because the budget is a target and not a floor. The seeded order is fixed
    when the first sweep starts and saved with the position. A later sweep
    reuses the order, so a prompt that joined the pool waits at the end. A
    prompt promoted in an earlier step leaves the pool, so its slot in the order
    is skipped. No prompt runs twice, and no remaining parked prompt is skipped.

    The schedule keeps a cycle position. Position `0` to `m - 1` is a sweep
    step and the rest of the cycle is a training step. The sweep length is fixed
    when the sweep starts and kept until the sweep ends, so a promotion in the
    middle of a sweep cannot shorten the cycle and skip its last block. The next
    sweep recomputes the length from the live parked count. The position is saved
    beside the label state after every step, so a resume continues the same
    sweep at the same index even when the parked count changed.

    The gap steps draw their train prompts from a train queue. The queue covers
    the whole train pool once before any prompt repeats. A promoted id joins the
    back of the queue. A demoted id leaves the queue and joins the end of the
    sweep order, so the next sweep runs it after every other parked prompt.
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
    _sweep_steps: int | None = field(default=None, init=False, repr=False)
    _train_queue: PromptQueue = field(
        default_factory=PromptQueue, init=False, repr=False
    )
    _train_queue_built: bool = field(default=False, init=False, repr=False)

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

    def planned_parked_share(self, parked: int) -> float:
        """The parked share of a step, averaged over one sweep cycle.

        A cycle presents `parked` prompts across its sweep steps, so the average
        parked prompts per step is the parked count divided by the cycle length.
        """
        cycle = self.cycle_length(parked)
        if cycle == 0:
            return 0.0
        return parked / (cycle * self.prompts_per_step)

    def _ensure_train_queue(self, train_ids: Sequence[str]) -> None:
        """Build the train queue from the pool on the first step."""
        if self._train_queue_built:
            return
        self._train_queue.build(train_ids, self.seed)
        self._train_queue_built = True

    def _prepare_order(self, parked_ids: Sequence[str]) -> None:
        """Build or repair the parked rotation for the current pool.

        The order is shuffled once. A later sweep reuses the rotation, so ids
        that left the pool are dropped and ids that joined it are appended in
        sorted order. A demoted id was already appended by `observe`, so it
        keeps its place at the end of the order.

        The repair runs at the start of a sweep only. Inside a sweep the order
        stays fixed, so a promoted id keeps its slot. A promoted id is filtered
        out when the block is built, and the prompts after it keep their
        positions. That is why no prompt repeats and no prompt is skipped.
        """
        if self._order is None:
            order = list(parked_ids)
            random.Random(self.seed).shuffle(order)
        else:
            order = list(self._order)
        present = set(parked_ids)
        kept: list[str] = []
        seen: set[str] = set()
        for prompt_id in order:
            if prompt_id in present and prompt_id not in seen:
                kept.append(prompt_id)
                seen.add(prompt_id)
        for prompt_id in sorted(present):
            if prompt_id not in seen:
                kept.append(prompt_id)
                seen.add(prompt_id)
        self._order = tuple(kept)

    def _train_step(
        self, train_by_id: Mapping[str, "Prompt"]
    ) -> tuple["Prompt", ...]:
        """One training block from the train queue."""
        block = self._train_queue.take(self.prompts_per_step)
        return tuple(train_by_id[prompt_id] for prompt_id in block)

    def plan(
        self,
        step: int,
        train_pool: Sequence["Prompt"],
        parked_pool: Sequence["Prompt"],
    ) -> StepPlan:
        """The prompts one step runs, from the saved cycle position.

        With no parked prompts the sweep never starts, the position stays at
        zero, and every step trains normally. A sweep step plans parked prompts
        only. A gap step plans train prompts only, taken from the train queue.
        """
        _positive_int("step", step)
        parked = sorted(parked_pool, key=_prompt_id)
        train = sorted(train_pool, key=_prompt_id)
        parked_by_id = {_prompt_id(prompt): prompt for prompt in parked}
        train_by_id = {_prompt_id(prompt): prompt for prompt in train}

        self._ensure_train_queue(list(train_by_id))
        _keep_and_fill(self._train_queue, list(train_by_id))

        sweep_steps = self.sweep_length(len(parked_by_id))
        if sweep_steps == 0:
            self._position = 0
            self._order = None
            self._sweep_steps = None
            return StepPlan(train=self._train_step(train_by_id))

        # The length is fixed when the sweep starts. A promotion during the
        # sweep shrinks the parked pool, and a live count would shorten the
        # cycle and skip the last block. The next sweep at position zero
        # recomputes the length.
        if self._position == 0 or self._sweep_steps is None:
            self._sweep_steps = sweep_steps
        sweep_steps = self._sweep_steps

        cycle = sweep_steps + self.gap_ratio * sweep_steps
        self._position %= cycle
        position = self._position

        if position < sweep_steps:
            if position == 0 or self._order is None:
                self._prepare_order(list(parked_by_id))
            start = position * self.prompts_per_step
            block = self._order[start : start + self.prompts_per_step]
            parked_slice = tuple(
                parked_by_id[prompt_id]
                for prompt_id in block
                if prompt_id in parked_by_id
            )
            self._position = (position + 1) % cycle
            return StepPlan(parked=parked_slice)

        self._position = (position + 1) % cycle
        return StepPlan(train=self._train_step(train_by_id))

    def observe(
        self, step: int, outcomes: Sequence[GroupOutcome]
    ) -> tuple[LabelUpdate, ...]:
        """Apply the promotion and demotion rules to one step's groups.

        A parked group is decided in the step it runs, so a sweep promotion does
        not wait for the rest of the sweep and a resume loses nothing. A train
        group in a gap step follows the spread demotion rule.

        A promoted id joins the back of the train queue. Its sweep slot stays in
        the order so the positions in the rest of the sweep do not move, and the
        slot is skipped when a block is built. A demoted id leaves the train
        queue and joins the end of the sweep order, unless the order already
        holds it. The position is clamped after a move, because the cycle length
        follows the parked count.
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
                self._move_in_queues(update)

        if updates:
            self._clamp_position()
        return tuple(updates)

    def _move_in_queues(self, update: LabelUpdate) -> None:
        """Move one changed id in the sweep order and the train queue."""
        if not self._train_queue_built:
            return
        if update.label == TRAIN_LABEL:
            self._train_queue.append(update.prompt_id)
            return
        self._train_queue.remove(update.prompt_id)
        if self._order is None or update.prompt_id in self._order:
            return
        self._order = (*self._order, update.prompt_id)

    def _clamp_position(self) -> None:
        """Keep the cycle position inside the cycle after a pool change."""
        if self._order is None:
            return
        sweep_steps = self._sweep_steps
        if sweep_steps is None:
            sweep_steps = self.sweep_length(len(self._order))
        cycle = sweep_steps + self.gap_ratio * sweep_steps
        if cycle:
            self._position %= cycle

    def state_row(self, step: int) -> dict[str, Any]:
        """The saved cycle position, frozen length and train queue after a step.

        The row also saves the sweep order. A resume then reuses the order from
        the sweep start, so a prompt promoted in an earlier step cannot run
        twice and no prompt is skipped. The frozen sweep length is saved too, so
        a resume mid-sweep keeps the same number of sweep steps. The parked list
        is the same order, because the shared resume path restores the train
        queue only from a row that holds both queue keys. `restore_order`
        remains the source for the parked steps.
        """
        row: dict[str, Any] = {"step": step, "sweep_position": self._position}
        if self._order is not None:
            row["sweep_order"] = list(self._order)
        if self._sweep_steps is not None:
            row["sweep_steps"] = self._sweep_steps
        row["parked_queue"] = list(self._order) if self._order is not None else []
        row["train_queue"] = self._train_queue.as_list()
        return row

    def restore_state(
        self, parked_queue: PromptQueue, train_queue: PromptQueue
    ) -> None:
        """Set the train queue from a saved `queues` row.

        The sweep does not draw parked prompts from a queue. The parked list in
        the row is the sweep order, which `restore_order` restores. The train
        queue is the source of rotation for the gap steps, so a resume
        continues the same train order.
        """
        if not isinstance(parked_queue, PromptQueue) or not isinstance(
            train_queue, PromptQueue
        ):
            raise TypeError("restore_state needs two PromptQueue objects")
        self._train_queue = train_queue
        self._train_queue_built = True

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

    def restore_sweep_steps(self, sweep_steps: int) -> None:
        """Set the frozen sweep length from a saved label row.

        A non-integer value or a value below one is ignored, because a
        hand-edited row must not break the run.
        """
        if isinstance(sweep_steps, bool) or not isinstance(sweep_steps, int):
            return
        if sweep_steps < 1:
            return
        self._sweep_steps = sweep_steps

    def failure_streak(self, prompt_id: str) -> int:
        """The current zero-pass count for a train prompt."""
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


def sweep_steps_from_row(row: Mapping[str, Any]) -> int | None:
    """The saved frozen sweep length, or None when the row has none."""
    sweep_steps = row.get("sweep_steps")
    if isinstance(sweep_steps, bool) or not isinstance(sweep_steps, int):
        return None
    if sweep_steps < 1:
        return None
    return sweep_steps


def sweep_order_from_row(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The saved sweep order, or None when the row has none or it is malformed."""
    order = row.get("sweep_order")
    if not isinstance(order, (list, tuple)) or not order:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in order):
        return None
    return tuple(order)
