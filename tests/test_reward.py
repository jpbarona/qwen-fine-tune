"""Tests for reward composition. The table is frozen in SPEC.md.

Task 4 implements the code-score arm only. The other three arms are named and
must fail clearly until their tasks land.
"""

from __future__ import annotations

import pytest

from rlvr.reward import (
    ARM_REWARDS,
    ARMS,
    ArmNotImplemented,
    RewardError,
    all_tests_passed,
    check_arm_implemented,
    code_score,
    compose_reward,
    reward_code_score,
    uses_output_judge,
    uses_process_judge,
)


def test_the_four_arm_names_are_frozen() -> None:
    assert ARMS == ("code-score", "hybrid", "gated-judge", "gated-process")


def test_only_code_score_avoids_the_output_judge() -> None:
    assert uses_output_judge("code-score") is False
    for arm in ("hybrid", "gated-judge", "gated-process"):
        assert uses_output_judge(arm) is True


def test_only_gated_process_uses_the_process_judge() -> None:
    assert uses_process_judge("gated-process") is True
    for arm in ("code-score", "hybrid", "gated-judge"):
        assert uses_process_judge(arm) is False


def test_code_score_is_the_fraction_of_tests_passed() -> None:
    assert code_score(1, 4) == 0.25
    assert code_score(3, 4) == 0.75
    assert code_score(0, 4) == 0.0
    assert code_score(4, 4) == 1.0


def test_code_score_reward_is_the_fraction() -> None:
    assert reward_code_score(tests_passed=2, tests_total=4) == 0.5
    assert compose_reward("code-score", tests_passed=2, tests_total=4) == 0.5
    assert compose_reward("code-score", tests_passed=4, tests_total=4) == 1.0
    assert compose_reward("code-score", tests_passed=0, tests_total=4) == 0.0


def test_code_score_reward_never_uses_a_judge() -> None:
    assert (
        reward_code_score(
            tests_passed=1,
            tests_total=2,
            judge_score=1.0,
            process_score=1.0,
        )
        == 0.5
    )


def test_a_rollout_with_zero_tests_fails_loudly() -> None:
    with pytest.raises(RewardError, match="zero tests"):
        code_score(0, 0)
    with pytest.raises(RewardError, match="zero tests"):
        compose_reward("code-score", tests_passed=0, tests_total=0)


def test_zero_tests_still_close_the_gate() -> None:
    assert all_tests_passed(0, 0) is False


def test_counts_must_be_sane() -> None:
    with pytest.raises(RewardError):
        code_score(3, 2)
    with pytest.raises(RewardError):
        code_score(-1, 2)


def test_only_code_score_is_implemented() -> None:
    assert set(ARM_REWARDS) == {"code-score"}
    assert set(ARM_REWARDS) | {"hybrid", "gated-judge", "gated-process"} == set(ARMS)
    check_arm_implemented("code-score")


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge", "gated-process"])
def test_the_other_three_arms_fail_clearly(arm: str) -> None:
    with pytest.raises(ArmNotImplemented, match=arm):
        check_arm_implemented(arm)
    with pytest.raises(ArmNotImplemented, match=arm):
        compose_reward(
            arm,
            tests_passed=4,
            tests_total=4,
            judge_score=1.0,
            process_score=1.0,
        )


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(RewardError, match="arm"):
        compose_reward("reward-shaping", tests_passed=1, tests_total=1)
