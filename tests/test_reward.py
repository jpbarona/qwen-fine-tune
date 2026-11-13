"""Tests for reward composition. The table is frozen in SPEC.md.

All four arms are implemented. The gated-process arm gates on the tests like
the other judge arms, then splits the reward evenly between the trace score and
the code score.
"""

from __future__ import annotations

import pytest

from rlvr.reward import (
    ARM_REWARDS,
    ARMS,
    PENDING_ARMS,
    RewardError,
    all_tests_passed,
    check_arm_implemented,
    code_score,
    compose_reward,
    gates_on_tests,
    reward_code_score,
    reward_gated_judge,
    reward_gated_process,
    reward_hybrid,
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


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # The plain test fraction.
        ({"tests_passed": 4, "tests_total": 4}, 1.0),
        ({"tests_passed": 3, "tests_total": 4}, 0.75),
        ({"tests_passed": 1, "tests_total": 4}, 0.25),
        ({"tests_passed": 0, "tests_total": 4}, 0.0),
        # Fixed penalties for truncation and code that does not compile.
        ({"tests_passed": 0, "tests_total": 4, "truncated": True}, -0.6),
        (
            {"tests_passed": 0, "tests_total": 4, "error_class": "syntax_error"},
            -0.5,
        ),
        (
            {"tests_passed": 0, "tests_total": 0, "error_class": "syntax_error"},
            -0.5,
        ),
        # The length term starts above 900 tokens and is capped at 0.2.
        ({"tests_passed": 4, "tests_total": 4, "completion_tokens": 900}, 1.0),
        ({"tests_passed": 4, "tests_total": 4, "completion_tokens": 1000}, 0.9),
        ({"tests_passed": 4, "tests_total": 4, "completion_tokens": 1100}, 0.8),
        ({"tests_passed": 4, "tests_total": 4, "completion_tokens": 4096}, 0.8),
        # The term applies to the whole output, whatever the base score.
        ({"tests_passed": 2, "tests_total": 4, "completion_tokens": 1100}, 0.3),
        ({"tests_passed": 0, "tests_total": 4, "completion_tokens": 1100}, -0.2),
        (
            {
                "tests_passed": 0,
                "tests_total": 4,
                "truncated": True,
                "completion_tokens": 1100,
            },
            -0.8,
        ),
        (
            {
                "tests_passed": 0,
                "tests_total": 4,
                "error_class": "syntax_error",
                "completion_tokens": 1100,
            },
            -0.7,
        ),
    ],
)
def test_the_code_arm_scores_each_case(kwargs: dict, expected: float) -> None:
    assert compose_reward("code-score", **kwargs) == pytest.approx(expected)
    assert reward_code_score(**kwargs) == pytest.approx(expected)


def test_the_code_arm_reward_is_clipped_to_the_range() -> None:
    assert compose_reward(
        "code-score",
        tests_passed=0,
        tests_total=4,
        truncated=True,
        completion_tokens=4096,
        length_penalty_floor=1.0,
    ) == -1.0
    assert compose_reward(
        "code-score",
        tests_passed=4,
        tests_total=4,
        completion_tokens=4096,
    ) == pytest.approx(0.8)


def test_the_length_term_uses_the_given_settings() -> None:
    assert compose_reward(
        "code-score",
        tests_passed=4,
        tests_total=4,
        completion_tokens=50,
        length_soft_limit=40,
        length_penalty_per_token=0.01,
        length_penalty_floor=0.05,
    ) == pytest.approx(0.95)


def test_any_correct_answer_beats_any_wrong_answer_at_every_length() -> None:
    correct = [
        compose_reward(
            "code-score", tests_passed=4, tests_total=4, completion_tokens=length
        )
        for length in range(0, 4097)
    ]
    wrong = [
        compose_reward(
            "code-score", tests_passed=0, tests_total=4, completion_tokens=length
        )
        for length in range(0, 4097)
    ]
    assert min(correct) == pytest.approx(0.8)
    assert max(wrong) == pytest.approx(0.0)
    assert min(correct) > max(wrong)


@pytest.mark.parametrize(
    "arm, scores",
    [
        ("hybrid", {"judge_score": 0.5}),
        ("gated-judge", {"judge_score": 0.5}),
        ("gated-process", {"judge_score": 0.5, "process_score": 0.5}),
    ],
)
def test_the_other_arms_ignore_the_rollout_context(
    arm: str, scores: dict[str, float]
) -> None:
    plain = compose_reward(arm, tests_passed=4, tests_total=4, **scores)
    contextual = compose_reward(
        arm,
        tests_passed=4,
        tests_total=4,
        truncated=True,
        error_class="syntax_error",
        completion_tokens=4096,
        **scores,
    )
    assert contextual == plain


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


def test_the_reward_table_is_the_single_arm_list() -> None:
    assert set(ARM_REWARDS) == set(ARMS)
    assert set(ARM_REWARDS) | PENDING_ARMS == set(ARMS)
    assert PENDING_ARMS == frozenset()
    for arm in ARM_REWARDS:
        check_arm_implemented(arm)


def test_judge_arms_gate_on_the_tests() -> None:
    assert gates_on_tests("code-score") is False
    for arm in ("hybrid", "gated-judge", "gated-process"):
        assert gates_on_tests(arm) is True


@pytest.mark.parametrize("judge", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_hybrid_on_a_full_pass_is_half_plus_half_judge(judge: float) -> None:
    expected = 0.5 + 0.5 * judge
    assert reward_hybrid(
        tests_passed=4, tests_total=4, judge_score=judge
    ) == pytest.approx(expected)
    assert compose_reward(
        "hybrid", tests_passed=4, tests_total=4, judge_score=judge
    ) == pytest.approx(expected)


@pytest.mark.parametrize("judge", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_gated_judge_on_a_full_pass_is_the_judge_score(judge: float) -> None:
    assert reward_gated_judge(
        tests_passed=2, tests_total=2, judge_score=judge
    ) == pytest.approx(judge)
    assert compose_reward(
        "gated-judge", tests_passed=2, tests_total=2, judge_score=judge
    ) == pytest.approx(judge)


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
@pytest.mark.parametrize(
    "passed,total",
    [(0, 4), (1, 4), (3, 4), (0, 0)],
)
def test_gated_arms_score_zero_when_the_gate_is_closed(
    arm: str, passed: int, total: int
) -> None:
    # No judge score is supplied, which proves the gate closes before the
    # judge is consulted.
    assert compose_reward(arm, tests_passed=passed, tests_total=total) == 0.0


def test_a_pass_without_a_judge_score_fails_loudly() -> None:
    for arm in ("hybrid", "gated-judge"):
        with pytest.raises(RewardError, match="judge"):
            compose_reward(arm, tests_passed=4, tests_total=4)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_a_judge_score_outside_the_unit_range_fails_loudly(bad: float) -> None:
    for arm in ("hybrid", "gated-judge"):
        with pytest.raises(RewardError, match="judge"):
            compose_reward(
                arm, tests_passed=4, tests_total=4, judge_score=bad
            )
    with pytest.raises(RewardError, match="judge"):
        compose_reward(
            "gated-process",
            tests_passed=4,
            tests_total=4,
            judge_score=bad,
            process_score=1.0,
        )


def test_a_boolean_judge_score_is_not_a_number() -> None:
    with pytest.raises(RewardError, match="judge"):
        compose_reward("hybrid", tests_passed=4, tests_total=4, judge_score=True)


def test_the_process_score_is_ignored_by_the_hybrid_arm() -> None:
    assert reward_hybrid(
        tests_passed=4,
        tests_total=4,
        judge_score=1.0,
        process_score=0.0,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("judge", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("process", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_gated_process_on_a_full_pass_is_half_process_plus_half_judge(
    judge: float, process: float
) -> None:
    expected = 0.5 * process + 0.5 * judge
    assert reward_gated_process(
        tests_passed=2,
        tests_total=2,
        judge_score=judge,
        process_score=process,
    ) == pytest.approx(expected)
    assert compose_reward(
        "gated-process",
        tests_passed=2,
        tests_total=2,
        judge_score=judge,
        process_score=process,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("passed,total", [(0, 4), (1, 4), (3, 4), (0, 0)])
def test_gated_process_scores_zero_when_the_gate_is_closed(
    passed: int, total: int
) -> None:
    # No scores are supplied, which proves the gate closes before either
    # judge score is consulted.
    assert compose_reward(
        "gated-process", tests_passed=passed, tests_total=total
    ) == 0.0


def test_a_process_pass_without_a_process_score_fails_loudly() -> None:
    with pytest.raises(RewardError, match="process"):
        compose_reward(
            "gated-process",
            tests_passed=4,
            tests_total=4,
            judge_score=1.0,
        )


def test_a_process_pass_without_a_judge_score_fails_loudly() -> None:
    with pytest.raises(RewardError, match="judge"):
        compose_reward(
            "gated-process",
            tests_passed=4,
            tests_total=4,
            process_score=1.0,
        )


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_a_process_score_outside_the_unit_range_fails_loudly(bad: float) -> None:
    with pytest.raises(RewardError, match="process"):
        compose_reward(
            "gated-process",
            tests_passed=4,
            tests_total=4,
            judge_score=1.0,
            process_score=bad,
        )


def test_a_boolean_process_score_is_not_a_number() -> None:
    with pytest.raises(RewardError, match="process"):
        compose_reward(
            "gated-process",
            tests_passed=4,
            tests_total=4,
            judge_score=1.0,
            process_score=True,
        )


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(RewardError, match="arm"):
        compose_reward("reward-shaping", tests_passed=1, tests_total=1)
