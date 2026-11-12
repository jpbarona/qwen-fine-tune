"""Tests for reward composition. The table is frozen in SPEC.md."""

from __future__ import annotations

import pytest

from rlvr.reward import (
    ARMS,
    code_score,
    compose_reward,
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
    assert compose_reward("code-score", tests_passed=2, tests_total=4) == 0.5
    assert compose_reward("code-score", tests_passed=4, tests_total=4) == 1.0
    assert compose_reward("code-score", tests_passed=0, tests_total=4) == 0.0


def test_no_tests_counts_as_a_failure() -> None:
    assert code_score(0, 0) == 0.0
    assert compose_reward("hybrid", tests_passed=0, tests_total=0, judge_score=0.9) == 0.0


def test_any_test_failure_gives_zero_for_gated_and_hybrid() -> None:
    for arm in ("hybrid", "gated-judge", "gated-process"):
        assert compose_reward(arm, tests_passed=3, tests_total=4, judge_score=1.0, process_score=1.0) == 0.0


def test_hybrid_on_full_pass_is_half_plus_half_judge() -> None:
    assert compose_reward("hybrid", tests_passed=4, tests_total=4, judge_score=0.6) == 0.8
    assert compose_reward("hybrid", tests_passed=4, tests_total=4, judge_score=0.0) == 0.5
    assert compose_reward("hybrid", tests_passed=4, tests_total=4, judge_score=1.0) == 1.0


def test_gated_judge_on_full_pass_is_the_judge_score() -> None:
    assert compose_reward("gated-judge", tests_passed=4, tests_total=4, judge_score=0.7) == 0.7


def test_gated_process_on_full_pass_mixes_process_and_judge() -> None:
    assert compose_reward(
        "gated-process", tests_passed=4, tests_total=4, judge_score=0.8, process_score=0.4
    ) == pytest.approx(0.6)


def test_a_required_score_is_demanded_when_tests_pass() -> None:
    from rlvr.reward import RewardError

    with pytest.raises(RewardError, match="judge"):
        compose_reward("hybrid", tests_passed=4, tests_total=4)
    with pytest.raises(RewardError, match="judge"):
        compose_reward("gated-judge", tests_passed=4, tests_total=4)
    with pytest.raises(RewardError, match="process"):
        compose_reward(
            "gated-process", tests_passed=4, tests_total=4, judge_score=0.5
        )


def test_scores_must_be_in_the_unit_interval() -> None:
    from rlvr.reward import RewardError

    with pytest.raises(RewardError):
        compose_reward("gated-judge", tests_passed=4, tests_total=4, judge_score=1.4)


def test_unknown_arm_is_rejected() -> None:
    from rlvr.reward import RewardError

    with pytest.raises(RewardError, match="arm"):
        compose_reward("reward-shaping", tests_passed=1, tests_total=1)
