"""Tests for the step schedule and the spread policy."""

from __future__ import annotations

import pytest

from rlvr.experiment import Prompt
from rlvr.queues import queues_from_row, restore_queue
from rlvr.schedule import (
    DEFAULT_DEMOTE_AFTER,
    DEFAULT_PROMOTION_BAR,
    GroupOutcome,
    LabelUpdate,
    Spread,
    Sweep,
    sweep_order_from_row,
    sweep_position_from_row,
    sweep_steps_from_row,
)

STEP = 38


def prompts(prefix: str, count: int) -> list[Prompt]:
    return [
        Prompt(prompt_id=f"{prefix}-{index:04d}", problem=f"problem {index}")
        for index in range(count)
    ]


def spread(**overrides) -> Spread:
    values = {"prompts_per_step": 19, "seed": 11}
    values.update(overrides)
    return Spread(**values)


@pytest.mark.parametrize(
    "prompts_per_step, expected",
    [
        (1, 0),
        (2, 1),
        (5, 4),
        (8, 4),
        (10, 5),
        (19, 9),
        (38, 19),
    ],
)
def test_the_parked_block_is_half_the_step_with_a_minimum_of_four(
    prompts_per_step: int, expected: int
) -> None:
    policy = Spread(prompts_per_step=prompts_per_step)

    assert policy.parked_count() == expected
    assert policy.parked_count() <= prompts_per_step - 1


def test_a_step_of_nineteen_prompts_has_nine_parked_and_ten_train() -> None:
    policy = spread()
    plan = policy.plan(1, prompts("train", 40), prompts("parked", 40))

    assert len(plan.parked) == 9
    assert len(plan.train) == 10
    assert len(plan) == 19


def test_the_planned_share_is_the_parked_block_over_the_step() -> None:
    assert spread().planned_parked_share() == pytest.approx(9 / 19)


@pytest.mark.parametrize(
    "prompts_per_step, parked, train",
    [(5, 4, 1), (2, 1, 1), (1, 0, 1)],
)
def test_a_small_step_keeps_one_train_prompt(
    prompts_per_step: int, parked: int, train: int
) -> None:
    policy = Spread(prompts_per_step=prompts_per_step)

    plan = policy.plan(1, prompts("train", 5), prompts("parked", 5))

    assert len(plan.parked) == parked
    assert len(plan.train) == train


def test_the_parked_queue_shows_every_prompt_before_a_repeat() -> None:
    policy = spread()
    parked = prompts("parked", 18)

    seen: list[str] = []
    for step in range(1, 3):
        seen.extend(
            prompt.prompt_id for prompt in policy.plan(step, [], parked).parked
        )

    assert len(seen) == 18
    assert len(set(seen)) == 18
    assert set(seen) == {prompt.prompt_id for prompt in parked}


def test_a_parked_pool_smaller_than_the_block_does_not_repeat() -> None:
    policy = spread()
    parked = prompts("parked", 3)

    block = policy.plan(1, prompts("train", 10), parked).parked

    assert len(block) == 3
    assert len({prompt.prompt_id for prompt in block}) == 3


def test_the_train_queue_shows_every_prompt_before_a_repeat() -> None:
    policy = spread()
    train = prompts("train", 20)
    parked = prompts("parked", 9)

    seen: list[str] = []
    for step in range(1, 3):
        seen.extend(
            prompt.prompt_id for prompt in policy.plan(step, train, parked).train
        )

    assert len(seen) == 20
    assert len(set(seen)) == 20
    assert set(seen) == {prompt.prompt_id for prompt in train}


def test_an_empty_parked_pool_leaves_the_whole_step_to_train() -> None:
    policy = spread()
    plan = policy.plan(1, prompts("train", 40), [])

    assert plan.parked == ()
    assert len(plan.train) == 19


def test_a_promotion_leaves_the_parked_queue_and_joins_the_back_of_train() -> None:
    policy = spread(prompts_per_step=5, seed=1)
    train = prompts("train", 3)
    parked = prompts("parked", 6)

    plan = policy.plan(1, train, parked)
    promoted = plan.parked[0].prompt_id

    updates = policy.observe(
        1, [GroupOutcome(prompt_id=promoted, passed=2, samples=16, parked=True)]
    )
    after = policy.state_row(1)

    assert [update.prompt_id for update in updates] == [promoted]
    assert promoted not in after["parked_queue"]
    assert after["train_queue"][-1] == promoted


def test_a_demotion_leaves_the_train_queue_and_joins_the_back_of_parked() -> None:
    policy = spread(prompts_per_step=5, seed=1, demote_after=2)
    train = prompts("train", 3)
    parked = prompts("parked", 6)

    plan = policy.plan(1, train, parked)
    demoted = plan.train[0].prompt_id
    failing = GroupOutcome(prompt_id=demoted, passed=0, samples=16, parked=False)

    assert policy.observe(1, [failing]) == ()
    updates = policy.observe(2, [failing])
    after = policy.state_row(2)

    assert [update.reason for update in updates] == ["demoted"]
    assert demoted not in after["train_queue"]
    assert after["parked_queue"][-1] == demoted


def test_a_resume_continues_the_same_queue_order() -> None:
    train = prompts("train", 40)
    parked = prompts("parked", 40)
    original = spread(seed=5)

    for step in range(1, 4):
        original.plan(step, train, parked)
    saved = queues_from_row(original.state_row(3))
    assert saved is not None
    parked_ids, train_ids = saved

    resumed = spread(seed=5)
    resumed.restore_state(
        restore_queue(parked_ids, (prompt.prompt_id for prompt in parked)),
        restore_queue(train_ids, (prompt.prompt_id for prompt in train)),
    )

    assert resumed.plan(4, train, parked) == original.plan(4, train, parked)


def test_a_queue_repairs_a_move_that_followed_the_saved_row() -> None:
    train = prompts("train", 3)
    parked = prompts("parked", 6)
    original = spread(prompts_per_step=5, seed=1)

    original.plan(1, train, parked)
    saved = queues_from_row(original.state_row(1))
    assert saved is not None
    parked_ids, train_ids = saved
    promoted = parked_ids[0]

    resumed = spread(prompts_per_step=5, seed=1)
    resumed.restore_state(
        restore_queue(parked_ids, (prompt.prompt_id for prompt in parked)),
        restore_queue(train_ids, (prompt.prompt_id for prompt in train)),
    )
    # A label row after the queues row moved the prompt in the live pool, but
    # the restored queue still holds the old order.
    moved = next(prompt for prompt in parked if prompt.prompt_id == promoted)
    new_train = train + [moved]
    new_parked = [prompt for prompt in parked if prompt.prompt_id != promoted]

    resumed.plan(2, new_train, new_parked)
    row = resumed.state_row(2)

    assert promoted not in row["parked_queue"]
    assert promoted in row["train_queue"]


def test_the_state_row_saves_both_queues() -> None:
    policy = spread()
    train = prompts("train", 12)
    parked = prompts("parked", 12)

    policy.plan(1, train, parked)
    row = policy.state_row(1)

    assert row["step"] == 1
    assert sorted(row["parked_queue"]) == [prompt.prompt_id for prompt in parked]
    assert sorted(row["train_queue"]) == [prompt.prompt_id for prompt in train]
    assert queues_from_row(row) is not None


def test_restore_state_needs_two_queues() -> None:
    with pytest.raises(TypeError, match="PromptQueue"):
        spread().restore_state([], [])


def test_the_seed_changes_the_queue_order() -> None:
    parked = prompts("parked", 100)
    train = prompts("train", 100)

    first = spread(seed=1).plan(1, train, parked)
    second = spread(seed=2).plan(1, train, parked)

    assert first != second


def test_the_prompt_order_does_not_change_the_plan() -> None:
    parked = prompts("parked", 40)
    train = prompts("train", 40)

    forward = spread(seed=5).plan(1, train, parked)
    backward = spread(seed=5).plan(
        1, list(reversed(train)), list(reversed(parked))
    )

    assert forward == backward


@pytest.mark.parametrize(
    "field, value",
    [
        ("prompts_per_step", 0),
        ("promotion_bar", 0),
        ("demote_after", 0),
        ("seed", False),
    ],
)
def test_bad_constructor_values_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        spread(**{field: value})


def test_a_plan_needs_a_positive_step() -> None:
    with pytest.raises(ValueError, match="step"):
        spread().plan(0, prompts("train", 1), [])


def test_promotion_needs_two_passes() -> None:
    policy = spread()
    one = GroupOutcome(prompt_id="p", passed=1, samples=16, parked=True)
    two = GroupOutcome(prompt_id="p", passed=2, samples=16, parked=True)

    assert policy.observe(1, [one]) == ()
    updates = policy.observe(1, [two])

    assert updates == (
        LabelUpdate(
            step=1,
            prompt_id="p",
            label="train",
            reason="promoted",
            passed=2,
            samples=16,
        ),
    )


def test_promotion_ignores_a_train_group() -> None:
    policy = spread()
    outcome = GroupOutcome(prompt_id="p", passed=16, samples=16, parked=False)

    assert policy.observe(1, [outcome]) == ()


def test_a_parked_full_failure_does_not_demote() -> None:
    policy = spread()
    outcome = GroupOutcome(prompt_id="p", passed=0, samples=16, parked=True)

    for step in range(1, 20):
        assert policy.observe(step, [outcome]) == ()


def test_demotion_needs_three_zero_pass_groups() -> None:
    policy = spread()
    outcome = GroupOutcome(prompt_id="p", passed=0, samples=16, parked=False)

    for step in range(1, DEFAULT_DEMOTE_AFTER):
        assert policy.observe(step, [outcome]) == ()
    assert DEFAULT_DEMOTE_AFTER == 3
    assert policy.failure_streak("p") == DEFAULT_DEMOTE_AFTER - 1

    updates = policy.observe(DEFAULT_DEMOTE_AFTER, [outcome])
    assert [update.label for update in updates] == ["parked"]
    assert updates[0].reason == "demoted"
    assert policy.failure_streak("p") == 0


def test_a_pass_does_not_clear_the_zero_pass_count() -> None:
    policy = spread()
    failing = GroupOutcome(prompt_id="p", passed=0, samples=16, parked=False)
    passing = GroupOutcome(prompt_id="p", passed=1, samples=16, parked=False)

    assert policy.observe(1, [failing]) == ()
    assert policy.observe(2, [failing]) == ()
    assert policy.observe(3, [passing]) == ()
    assert policy.failure_streak("p") == 2

    updates = policy.observe(4, [failing])
    assert [update.reason for update in updates] == ["demoted"]
    assert policy.failure_streak("p") == 0


def test_a_restored_count_waits_for_a_zero_pass_group() -> None:
    policy = spread()
    passing = GroupOutcome(prompt_id="p", passed=1, samples=16, parked=False)
    failing = GroupOutcome(prompt_id="p", passed=0, samples=16, parked=False)

    policy.restore_streak("p", 3)

    assert policy.observe(1, [passing]) == ()
    assert policy.failure_streak("p") == 3

    updates = policy.observe(2, [failing])
    assert [update.reason for update in updates] == ["demoted"]
    assert policy.failure_streak("p") == 0


def test_separate_prompts_keep_separate_counts() -> None:
    policy = spread(demote_after=10)
    first = GroupOutcome(prompt_id="a", passed=0, samples=16, parked=False)
    second = GroupOutcome(prompt_id="b", passed=0, samples=16, parked=False)

    for step in range(1, 4):
        policy.observe(step, [first])
    assert policy.failure_streak("a") == 3
    assert policy.failure_streak("b") == 0

    policy.observe(4, [first, second])
    assert policy.failure_streak("a") == 4
    assert policy.failure_streak("b") == 1


def test_the_defaults_match_the_binding_values() -> None:
    assert DEFAULT_PROMOTION_BAR == 2
    assert DEFAULT_DEMOTE_AFTER == 3


def test_a_label_update_dict_has_the_run_file_fields() -> None:
    update = LabelUpdate(
        step=3,
        prompt_id="p",
        label="parked",
        reason="demoted",
        passed=0,
        samples=16,
    )

    assert update.to_dict() == {
        "step": 3,
        "prompt_id": "p",
        "label": "parked",
        "reason": "demoted",
        "passed": 0,
        "samples": 16,
    }


@pytest.mark.parametrize(
    "values",
    [
        {"prompt_id": "p", "passed": 17, "samples": 16, "parked": True},
        {"prompt_id": "p", "passed": -1, "samples": 16, "parked": True},
        {"prompt_id": "p", "passed": 0, "samples": 0, "parked": True},
        {"prompt_id": "", "passed": 0, "samples": 16, "parked": True},
    ],
)
def test_a_bad_group_outcome_is_rejected(values: dict) -> None:
    with pytest.raises(ValueError):
        GroupOutcome(**values)


def test_a_saved_streak_is_restored_and_zero_clears_it() -> None:
    policy = spread()

    policy.restore_streak("p", 7)
    assert policy.failure_streak("p") == 7

    policy.restore_streak("p", 0)
    assert policy.failure_streak("p") == 0


def test_a_bad_saved_streak_is_ignored() -> None:
    policy = spread()
    policy.restore_streak("p", 4)

    policy.restore_streak("p", -1)
    policy.restore_streak("p", True)
    assert policy.failure_streak("p") == 4


def test_streak_from_row_reads_a_saved_streak() -> None:
    from rlvr.schedule import streak_from_row

    assert streak_from_row({"prompt_id": "p", "streak": 9}) == ("p", 9)
    assert streak_from_row({"prompt_id": "p", "streak": -1}) is None
    assert streak_from_row({"prompt_id": "p", "label": "train"}) is None


def sweep(**overrides) -> Sweep:
    values = {
        "prompts_per_step": STEP,
        "gap_ratio": 3,
        "seed": 11,
    }
    values.update(overrides)
    return Sweep(**values)


def test_the_sweep_length_rounds_up_the_parked_count() -> None:
    policy = sweep(prompts_per_step=10)

    assert policy.sweep_length(0) == 0
    assert policy.sweep_length(1) == 1
    assert policy.sweep_length(10) == 1
    assert policy.sweep_length(11) == 2
    assert policy.sweep_length(35) == 4


def test_the_gap_is_the_ratio_times_the_sweep_length() -> None:
    policy = sweep(prompts_per_step=10, gap_ratio=3)

    assert policy.gap_length(35) == 12
    assert policy.cycle_length(35) == 16


def test_a_sweep_runs_all_parked_steps_then_the_gap() -> None:
    # Four parked prompts at two per step gives two sweep steps and a gap of
    # six, so the cycle is eight steps.
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    parked = prompts("parked", 4)
    train = prompts("train", 4)

    sweep_flags = [
        bool(policy.plan(step, train, parked).parked)
        for step in range(1, 10)
    ]

    assert sweep_flags == [True, True, False, False, False, False, False, False, True]


def test_a_sweep_step_holds_parked_prompts_only() -> None:
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    parked = prompts("parked", 4)
    train = prompts("train", 4)

    plan = policy.plan(1, train, parked)

    assert plan.train == ()
    assert len(plan.parked) == 2


def test_a_cycle_covers_every_parked_prompt_once() -> None:
    # Ten prompts at two per step is five sweep steps and a gap of fifteen.
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    parked = prompts("parked", 10)
    train = prompts("train", 5)

    seen: list[str] = []
    for step in range(1, 6):
        seen.extend(prompt.prompt_id for prompt in policy.plan(step, train, parked).parked)

    assert sorted(seen) == [prompt.prompt_id for prompt in parked]


def test_a_pool_smaller_than_one_step_does_not_repeat_a_prompt() -> None:
    policy = sweep(prompts_per_step=38, gap_ratio=3)
    parked = prompts("parked", 3)

    plan = policy.plan(1, prompts("train", 5), parked)

    assert plan.train == ()
    assert len(plan.parked) == 3
    assert {prompt.prompt_id for prompt in plan.parked} == {
        prompt.prompt_id for prompt in parked
    }
    assert policy.cycle_length(3) == 4


def test_a_short_final_sweep_step_holds_only_the_rest() -> None:
    # Five parked prompts at two per step gives three sweep steps, the last one
    # holding one prompt. No prompt is repeated in a step.
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    parked = prompts("parked", 5)
    train = prompts("train", 3)

    plans = [policy.plan(step, train, parked) for step in (1, 2, 3)]
    sizes = [len(plan.parked) for plan in plans]
    seen = [prompt.prompt_id for plan in plans for prompt in plan.parked]

    assert sizes == [2, 2, 1]
    assert sorted(seen) == [prompt.prompt_id for prompt in parked]


def test_a_promotion_mid_sweep_does_not_repeat_a_prompt() -> None:
    # Four parked prompts at two per step with seed 1 starts with p3 and p0.
    # When p0 promotes, the next step must run the next two prompts in the same
    # order, p2 and p1, not a fresh shuffle that can pick p0 again.
    policy = sweep(prompts_per_step=2, gap_ratio=3, seed=1)
    parked = prompts("parked", 4)

    first = policy.plan(1, [], parked)
    assert [prompt.prompt_id for prompt in first.parked] == [
        "parked-0003",
        "parked-0000",
    ]

    policy.observe(
        1,
        [GroupOutcome(prompt_id="parked-0000", passed=2, samples=16, parked=True)],
    )
    remaining = [prompt for prompt in parked if prompt.prompt_id != "parked-0000"]
    second = policy.plan(2, [], remaining)

    assert [prompt.prompt_id for prompt in second.parked] == [
        "parked-0002",
        "parked-0001",
    ]
    assert not {"parked-0003", "parked-0000"} & {
        prompt.prompt_id for prompt in second.parked
    }


def test_a_promotion_mid_sweep_does_not_skip_the_last_block() -> None:
    # Nine parked prompts at four per step gives three sweep steps. One
    # promotion in the first step leaves eight prompts, which is two steps, so
    # a live count would drop the last block and skip one prompt.
    policy = sweep(prompts_per_step=4, gap_ratio=3, seed=3)
    parked = prompts("parked", 9)
    train = prompts("train", 6)

    first = policy.plan(1, train, parked)
    assert len(first.parked) == 4
    promoted = first.parked[0].prompt_id
    updates = policy.observe(
        1,
        [GroupOutcome(prompt_id=promoted, passed=2, samples=16, parked=True)],
    )
    assert [update.reason for update in updates] == ["promoted"]
    remaining = [prompt for prompt in parked if prompt.prompt_id != promoted]

    seen = [prompt.prompt_id for prompt in first.parked]
    for step in (2, 3):
        plan = policy.plan(step, train, remaining)
        assert plan.train == ()
        seen.extend(prompt.prompt_id for prompt in plan.parked)

    gap = policy.plan(4, train, remaining)

    assert gap.parked == ()
    assert len(seen) == 9
    assert sorted(seen) == [prompt.prompt_id for prompt in parked]
    assert set(seen) == {prompt.prompt_id for prompt in parked}


def test_a_resume_mid_sweep_keeps_the_frozen_length() -> None:
    parked = prompts("parked", 9)
    train = prompts("train", 6)
    original = sweep(prompts_per_step=4, gap_ratio=3, seed=3)

    first = original.plan(1, train, parked)
    promoted = first.parked[0].prompt_id
    original.observe(
        1,
        [GroupOutcome(prompt_id=promoted, passed=2, samples=16, parked=True)],
    )
    saved = original.state_row(1)
    remaining = [prompt for prompt in parked if prompt.prompt_id != promoted]
    expected = [original.plan(step, train, remaining) for step in (2, 3)]

    order = sweep_order_from_row(saved)
    assert order is not None
    assert sweep_steps_from_row(saved) == 3

    resumed = sweep(prompts_per_step=4, gap_ratio=3, seed=3)
    resumed.restore_position(saved["sweep_position"])
    resumed.restore_order(order)
    resumed.restore_sweep_steps(sweep_steps_from_row(saved))

    assert [resumed.plan(step, train, remaining) for step in (2, 3)] == expected


def test_a_saved_sweep_order_keeps_the_sequence_across_a_resume() -> None:
    parked = prompts("parked", 4)
    original = sweep(prompts_per_step=2, gap_ratio=3, seed=1)

    original.plan(1, [], parked)
    original.observe(
        1,
        [GroupOutcome(prompt_id="parked-0000", passed=2, samples=16, parked=True)],
    )
    row = original.state_row(1)
    remaining = [prompt for prompt in parked if prompt.prompt_id != "parked-0000"]
    expected = original.plan(2, [], remaining)

    resumed = sweep(prompts_per_step=2, gap_ratio=3, seed=1)
    resumed.restore_position(row["sweep_position"])
    order = sweep_order_from_row(row)
    assert order is not None
    resumed.restore_order(order)

    assert resumed.plan(2, [], remaining) == expected


def test_an_empty_parked_pool_never_starts_a_sweep() -> None:
    policy = sweep()
    train = prompts("train", 50)

    first = policy.plan(1, train, [])
    second = policy.plan(2, train, [])

    assert first.parked == ()
    assert second.parked == ()
    assert len(first.train) == STEP
    assert len(second.train) == STEP
    assert policy.position == 0


def test_a_resume_mid_sweep_continues_at_the_saved_index() -> None:
    parked = prompts("parked", 4)
    train = prompts("train", 4)
    original = sweep(prompts_per_step=2, gap_ratio=3)

    original.plan(1, train, parked)
    saved = original.state_row(1)

    resumed = sweep(prompts_per_step=2, gap_ratio=3)
    resumed.restore_position(saved["sweep_position"])

    assert resumed.plan(2, train, parked) == original.plan(2, train, parked)


def test_a_resume_after_a_sweep_trains_in_the_gap() -> None:
    parked = prompts("parked", 4)
    train = prompts("train", 4)
    original = sweep(prompts_per_step=2, gap_ratio=3)

    for step in (1, 2):
        original.plan(step, train, parked)
    saved = original.state_row(2)

    resumed = sweep(prompts_per_step=2, gap_ratio=3)
    resumed.restore_position(saved["sweep_position"])
    plan = resumed.plan(3, train, parked)

    assert plan.parked == ()
    assert len(plan.train) == 2


def test_the_state_row_carries_the_next_position() -> None:
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    parked = prompts("parked", 4)

    policy.plan(1, prompts("train", 4), parked)
    row = policy.state_row(1)

    assert row["step"] == 1
    assert row["sweep_position"] == 1
    order = sweep_order_from_row(row)
    assert order is not None
    assert sorted(order) == [prompt.prompt_id for prompt in parked]
    assert sweep_position_from_row({"step": 1, "sweep_position": 1}) == 1
    assert sweep_position_from_row({"step": 1, "prompt_id": "p"}) is None
    assert sweep_position_from_row({"step": 1, "sweep_position": -1}) is None
    assert sweep_steps_from_row(row) == 2
    assert sweep_steps_from_row({"step": 1, "sweep_steps": 0}) is None
    assert sweep_steps_from_row({"step": 1, "sweep_steps": True}) is None
    assert sweep_steps_from_row({"step": 1}) is None


def test_a_bad_saved_position_is_ignored() -> None:
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    policy.restore_position(1)

    policy.restore_position(-1)
    policy.restore_position(True)
    assert policy.position == 1


def test_a_sweep_step_promotes_in_the_same_step() -> None:
    policy = sweep(prompts_per_step=1, gap_ratio=3)
    parked = prompts("parked", 2)
    passed = GroupOutcome(prompt_id="parked-0000", passed=2, samples=16, parked=True)

    policy.plan(1, [], parked)
    updates = policy.observe(1, [passed])

    assert [update.prompt_id for update in updates] == ["parked-0000"]
    assert updates[0].label == "train"
    assert updates[0].reason == "promoted"


def test_a_sweep_group_never_demotes() -> None:
    policy = sweep(prompts_per_step=1, gap_ratio=3)
    parked = prompts("parked", 1)
    failing = GroupOutcome(prompt_id="parked-0000", passed=0, samples=16, parked=True)

    policy.plan(1, [], parked)
    assert policy.observe(1, [failing]) == ()
    assert policy.failure_streak("parked-0000") == 0


def test_a_gap_step_demotes_a_full_failure_train_prompt() -> None:
    policy = sweep(prompts_per_step=1, gap_ratio=3, demote_after=1)
    parked = prompts("parked", 1)
    train = prompts("train", 1)
    failing = GroupOutcome(prompt_id="train-0000", passed=0, samples=16, parked=False)

    policy.plan(1, train, parked)
    policy.plan(2, train, parked)
    updates = policy.observe(2, [failing])

    assert [update.reason for update in updates] == ["demoted"]


def test_the_gap_covers_the_train_pool_before_a_repeat() -> None:
    # Four parked prompts at two per step is two sweep steps and a gap of six.
    # The train pool holds ten prompts, so five gap steps cover all of them.
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    parked = prompts("parked", 4)
    train = prompts("train", 10)

    policy.plan(1, train, parked)
    policy.plan(2, train, parked)
    seen: list[str] = []
    for step in range(3, 8):
        plan = policy.plan(step, train, parked)
        assert plan.parked == ()
        seen.extend(prompt.prompt_id for prompt in plan.train)

    assert len(seen) == 10
    assert len(set(seen)) == 10
    assert set(seen) == {prompt.prompt_id for prompt in train}


def test_a_promoted_id_joins_the_back_of_the_train_queue() -> None:
    policy = sweep(prompts_per_step=2, gap_ratio=3, seed=1)
    parked = prompts("parked", 4)
    train = prompts("train", 3)

    first = policy.plan(1, train, parked)
    promoted = first.parked[0].prompt_id
    updates = policy.observe(
        1, [GroupOutcome(prompt_id=promoted, passed=2, samples=16, parked=True)]
    )
    row = policy.state_row(1)

    assert [update.reason for update in updates] == ["promoted"]
    assert row["train_queue"][-1] == promoted


def test_a_demoted_id_leaves_training_and_joins_the_end_of_the_sweep_order() -> None:
    policy = sweep(prompts_per_step=2, gap_ratio=3, seed=1, demote_after=1)
    parked = prompts("parked", 4)
    train = prompts("train", 3)

    policy.plan(1, train, parked)
    policy.plan(2, train, parked)
    gap = policy.plan(3, train, parked)
    assert gap.parked == ()
    demoted = gap.train[0].prompt_id
    failing = GroupOutcome(prompt_id=demoted, passed=0, samples=16, parked=False)

    updates = policy.observe(3, [failing])
    row = policy.state_row(3)
    order = sweep_order_from_row(row)

    assert [update.reason for update in updates] == ["demoted"]
    assert demoted not in row["train_queue"]
    assert order is not None
    assert order[-1] == demoted


def test_the_sweep_state_row_saves_the_train_queue() -> None:
    policy = sweep(prompts_per_step=2, gap_ratio=3, seed=1)
    parked = prompts("parked", 4)
    train = prompts("train", 6)

    policy.plan(1, train, parked)
    row = policy.state_row(1)

    assert sorted(row["train_queue"]) == [prompt.prompt_id for prompt in train]
    assert queues_from_row(row) is not None


def test_a_resume_mid_gap_continues_the_same_train_order() -> None:
    parked = prompts("parked", 4)
    train = prompts("train", 10)
    original = sweep(prompts_per_step=2, gap_ratio=3, seed=5)

    for step in range(1, 5):
        original.plan(step, train, parked)
    saved = original.state_row(4)
    expected = original.plan(5, train, parked)

    saved_queues = queues_from_row(saved)
    assert saved_queues is not None
    parked_ids, train_ids = saved_queues
    order = sweep_order_from_row(saved)
    assert order is not None

    resumed = sweep(prompts_per_step=2, gap_ratio=3, seed=5)
    resumed.restore_position(saved["sweep_position"])
    resumed.restore_order(order)
    resumed.restore_state(
        restore_queue(parked_ids, (prompt.prompt_id for prompt in parked)),
        restore_queue(train_ids, (prompt.prompt_id for prompt in train)),
    )

    assert resumed.plan(5, train, parked) == expected


def test_a_sweep_restore_state_needs_two_queues() -> None:
    with pytest.raises(TypeError, match="PromptQueue"):
        sweep().restore_state([], [])


def test_the_spread_share_holds_while_the_parked_pool_fills_the_block() -> None:
    # At 38 prompts a step the parked block is 19 and the train block is 19.
    policy = Spread(prompts_per_step=STEP, seed=0)
    parked = prompts("parked", 100)
    train = prompts("train", 100)

    plans = [policy.plan(step, train, parked) for step in range(1, 5)]

    assert [len(plan.parked) for plan in plans] == [19, 19, 19, 19]
    assert [len(plan.train) for plan in plans] == [19, 19, 19, 19]
    assert policy.planned_parked_share() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "field, value",
    [
        ("prompts_per_step", 0),
        ("gap_ratio", 2),
        ("gap_ratio", 11),
        ("gap_ratio", 3.0),
        ("gap_ratio", True),
        ("promotion_bar", 0),
        ("demote_after", 0),
        ("seed", False),
    ],
)
def test_bad_sweep_constructor_values_are_rejected(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        sweep(**{field: value})
