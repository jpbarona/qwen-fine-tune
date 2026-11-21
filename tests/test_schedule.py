"""Tests for the step schedule and the spread policy."""

from __future__ import annotations

import math

import pytest

from rlvr.experiment import Prompt
from rlvr.schedule import (
    DEFAULT_DEMOTE_AFTER,
    DEFAULT_PROMOTION_BAR,
    GroupOutcome,
    LabelUpdate,
    Spread,
    Sweep,
    sweep_position_from_row,
)

STEP = 38
CAP = 9
REFRESH = 40


def prompts(prefix: str, count: int) -> list[Prompt]:
    return [
        Prompt(prompt_id=f"{prefix}-{index:04d}", problem=f"problem {index}")
        for index in range(count)
    ]


def spread(**overrides) -> Spread:
    values = {
        "prompts_per_step": STEP,
        "cap": CAP,
        "refresh_steps": REFRESH,
        "seed": 11,
    }
    values.update(overrides)
    return Spread(**values)


@pytest.mark.parametrize(
    "parked_count, expected_slice",
    [
        (0, 0),
        (9, 1),
        (300, 8),
        (1000, 9),
    ],
)
def test_the_parked_slice_is_the_cap_or_the_refresh_count(
    parked_count: int, expected_slice: int
) -> None:
    policy = spread()
    plan = policy.plan(
        1,
        prompts("train", 50),
        prompts("parked", parked_count),
    )

    expected = min(CAP, math.ceil(parked_count / REFRESH))
    assert expected == expected_slice
    assert len(plan.parked) == expected
    assert len(plan.train) == STEP - expected
    assert len(plan) == STEP


def test_the_cap_binds_when_the_refresh_count_is_larger() -> None:
    policy = spread(cap=3)
    plan = policy.plan(1, prompts("train", 50), prompts("parked", 1000))

    assert len(plan.parked) == 3


def test_a_parked_pool_smaller_than_one_step_still_takes_one_prompt() -> None:
    policy = spread()
    plan = policy.plan(1, prompts("train", 10), prompts("parked", 1))

    assert len(plan.parked) == 1
    assert len(plan.train) == STEP - 1


def test_train_prompts_cycle_when_the_pool_is_smaller_than_the_step() -> None:
    policy = spread(prompts_per_step=5, cap=1, refresh_steps=1)
    plan = policy.plan(1, prompts("train", 2), prompts("parked", 3))

    assert [prompt.prompt_id for prompt in plan.train] == [
        "train-0000",
        "train-0001",
        "train-0000",
        "train-0001",
    ]
    assert len(plan.parked) == 1


def test_an_empty_parked_pool_leaves_the_whole_step_to_train() -> None:
    policy = spread()
    plan = policy.plan(7, prompts("train", 50), [])

    assert plan.parked == ()
    assert len(plan.train) == STEP


def test_the_rotation_visits_every_parked_prompt_once_per_cycle() -> None:
    policy = spread(prompts_per_step=5, cap=2, refresh_steps=9)
    parked = prompts("parked", 9)

    seen: list[str] = []
    for step in range(1, 10):
        seen.extend(prompt.prompt_id for prompt in policy.plan(step, [], parked).parked)

    assert sorted(seen) == [prompt.prompt_id for prompt in parked]


def test_the_rotation_is_stable_across_a_resume() -> None:
    parked = prompts("parked", 300)
    train = prompts("train", 60)

    fresh = spread(seed=5)
    original = spread(seed=5)
    for step in range(1, 42):
        original.plan(step, train, parked)

    assert original.plan(41, train, parked) == fresh.plan(41, train, parked)


def test_the_seed_changes_the_rotation() -> None:
    parked = prompts("parked", 100)
    first = spread(seed=1).plan(1, [], parked)
    second = spread(seed=2).plan(1, [], parked)

    assert first != second


def test_the_prompt_order_does_not_change_the_plan() -> None:
    parked = prompts("parked", 40)
    train = prompts("train", 10)
    policy = spread(seed=5)

    forward = policy.plan(9, train, parked)
    backward = policy.plan(9, list(reversed(train)), list(reversed(parked)))

    assert forward == backward


@pytest.mark.parametrize(
    "field, value",
    [
        ("prompts_per_step", 0),
        ("refresh_steps", 0),
        ("cap", -1),
        ("promotion_bar", 0),
        ("demote_after", 0),
        ("cap", 100),
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


def test_demotion_needs_ten_consecutive_full_failure_groups() -> None:
    policy = spread()
    outcome = GroupOutcome(prompt_id="p", passed=0, samples=16, parked=False)

    for step in range(1, DEFAULT_DEMOTE_AFTER):
        assert policy.observe(step, [outcome]) == ()
    assert policy.failure_streak("p") == DEFAULT_DEMOTE_AFTER - 1

    updates = policy.observe(DEFAULT_DEMOTE_AFTER, [outcome])
    assert [update.label for update in updates] == ["parked"]
    assert updates[0].reason == "demoted"
    assert policy.failure_streak("p") == 0


def test_one_passing_rollout_resets_the_failure_streak() -> None:
    policy = spread()
    failing = GroupOutcome(prompt_id="p", passed=0, samples=16, parked=False)
    passing = GroupOutcome(prompt_id="p", passed=1, samples=16, parked=False)

    for step in range(1, 10):
        policy.observe(step, [failing])
    policy.observe(10, [passing])
    assert policy.failure_streak("p") == 0

    for step in range(11, 20):
        assert policy.observe(step, [failing]) == ()


def test_separate_prompts_keep_separate_streaks() -> None:
    policy = spread()
    first = GroupOutcome(prompt_id="a", passed=0, samples=16, parked=False)
    second = GroupOutcome(prompt_id="b", passed=0, samples=16, parked=False)

    for step in range(1, 10):
        policy.observe(step, [first])
    assert policy.failure_streak("a") == 9
    assert policy.failure_streak("b") == 0

    policy.observe(10, [first, second])
    assert policy.failure_streak("a") == 0
    assert policy.failure_streak("b") == 1


def test_the_defaults_match_the_binding_values() -> None:
    assert DEFAULT_PROMOTION_BAR == 2
    assert DEFAULT_DEMOTE_AFTER == 10


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


def test_a_pool_smaller_than_one_step_still_uses_the_whole_step() -> None:
    policy = sweep(prompts_per_step=38, gap_ratio=3)
    parked = prompts("parked", 3)

    plan = policy.plan(1, prompts("train", 5), parked)

    assert plan.train == ()
    assert len(plan.parked) == 38
    assert {prompt.prompt_id for prompt in plan.parked} == {
        prompt.prompt_id for prompt in parked
    }
    assert policy.cycle_length(3) == 4


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

    assert policy.state_row(1) == {"step": 1, "sweep_position": 1}
    assert sweep_position_from_row({"step": 1, "sweep_position": 1}) == 1
    assert sweep_position_from_row({"step": 1, "prompt_id": "p"}) is None
    assert sweep_position_from_row({"step": 1, "sweep_position": -1}) is None


def test_a_bad_saved_position_is_ignored() -> None:
    policy = sweep(prompts_per_step=2, gap_ratio=3)
    policy.restore_position(1)

    policy.restore_position(-1)
    policy.restore_position(True)
    assert policy.position == 1


def test_promotions_wait_for_the_end_of_the_sweep() -> None:
    # One parked prompt at one prompt per step is one sweep step.
    policy = sweep(prompts_per_step=1, gap_ratio=3)
    parked = prompts("parked", 2)
    passed = GroupOutcome(prompt_id="parked-0000", passed=2, samples=16, parked=True)

    policy.plan(1, [], parked)
    assert policy.observe(1, [passed]) == ()

    policy.plan(2, [], parked)
    updates = policy.observe(
        2,
        [GroupOutcome(prompt_id="parked-0001", passed=2, samples=16, parked=True)],
    )

    assert [update.prompt_id for update in updates] == ["parked-0000", "parked-0001"]
    assert all(update.label == "train" for update in updates)


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


def test_both_policies_spend_the_same_parked_share_within_one_prompt() -> None:
    # At the binding values the spread cap is nine prompts and the sweep gap is
    # three sweep lengths. Over one cycle the two shares differ by less than one
    # prompt's worth of a step.
    parked_count = 380
    parked = prompts("parked", parked_count)
    train = prompts("train", 50)
    spread = Spread(prompts_per_step=STEP, cap=CAP, refresh_steps=REFRESH, seed=0)
    sweep_policy = Sweep(prompts_per_step=STEP, gap_ratio=3, seed=0)

    cycle = sweep_policy.cycle_length(parked_count)
    spread_parked = sum(
        len(spread.plan(step, train, parked).parked) for step in range(1, cycle + 1)
    )
    sweep_parked = sum(
        len(sweep_policy.plan(step, train, parked).parked)
        for step in range(1, cycle + 1)
    )

    spread_share = spread_parked / (cycle * STEP)
    sweep_share = sweep_parked / (cycle * STEP)

    assert sweep_share == 1 / (1 + 3)
    assert abs(spread_share - sweep_share) <= 1 / STEP


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
