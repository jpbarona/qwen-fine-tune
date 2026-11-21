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
