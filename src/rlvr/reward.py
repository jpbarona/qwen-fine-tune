"""Reward composition for the four arms.

The table comes from SPEC.md and is frozen:

| Arm            | Tests fail               | Tests pass                  |
|----------------|--------------------------|-----------------------------|
| code-score     | fraction of tests passed | 1.0                         |
| hybrid         | 0                        | 0.5 + 0.5 * judge           |
| gated-judge    | 0                        | judge                       |
| gated-process  | 0                        | 0.5 * process + 0.5 * judge |

Each arm gets one small function, and `ARM_REWARDS` is the single dispatch
point. All four arms are implemented. Adding an arm means adding one entry to
`ARM_REWARDS`. The Experiment loop never changes.

"Tests fail" means any test failed, so a rollout that passes nothing and one
that passes all but one both count as failures for the gated arms. A task with
zero tests is an error for the code score: nothing was proven, and returning
0.0 would hide a broken test runner. The gated arms differ: a question with no
tests cannot pass the gate, so it gets zero and the judge is not called. The
Experiment records a code score of 0.0 for that case.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = [
    "ARM_REWARDS",
    "ARMS",
    "ArmNotImplemented",
    "PENDING_ARMS",
    "RewardError",
    "all_tests_passed",
    "check_arm_implemented",
    "code_score",
    "compose_reward",
    "gates_on_tests",
    "reward_code_score",
    "reward_gated_judge",
    "reward_gated_process",
    "reward_hybrid",
    "uses_output_judge",
    "uses_process_judge",
]

#: Arm names in the order used by the CLI and the run folders.
ARMS: tuple[str, ...] = ("code-score", "hybrid", "gated-judge", "gated-process")

#: Arms whose reward is composed from judge output.
OUTPUT_JUDGE_ARMS = frozenset({"hybrid", "gated-judge", "gated-process"})

#: Arms whose reward is composed from the process judge.
PROCESS_JUDGE_ARMS = frozenset({"gated-process"})

RewardFn = Callable[..., float]


class RewardError(ValueError):
    """A reward could not be composed, because an arm or a score is wrong."""


class ArmNotImplemented(NotImplementedError):
    """The arm has a name but no reward yet. A run must stop before it starts."""


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise RewardError(f"unknown arm {arm!r}. valid arms: {', '.join(ARMS)}")


def _check_counts(tests_passed: int, tests_total: int) -> None:
    if tests_total < 0 or tests_passed < 0:
        raise RewardError(
            f"test counts must not be negative, got passed={tests_passed} total={tests_total}"
        )
    if tests_passed > tests_total:
        raise RewardError(
            f"tests_passed must not exceed tests_total, got {tests_passed} > {tests_total}"
        )


def uses_output_judge(arm: str) -> bool:
    """Whether the arm asks the output judge for a score at all."""
    _check_arm(arm)
    return arm in OUTPUT_JUDGE_ARMS


def uses_process_judge(arm: str) -> bool:
    """Whether the arm asks the process judge for a score at all."""
    _check_arm(arm)
    return arm in PROCESS_JUDGE_ARMS


def gates_on_tests(arm: str) -> bool:
    """Whether the arm returns zero when any test fails.

    Every arm that calls the output judge also gates on a full test pass. The
    code-score arm has no gate: it returns the fraction of tests passed.
    """
    _check_arm(arm)
    return arm in OUTPUT_JUDGE_ARMS


def _check_score(name: str, score: float | None) -> float:
    """The judge score, checked to be a real number in [0, 1]."""
    if score is None:
        raise RewardError(f"this arm needs a {name} score on a full test pass, got null")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RewardError(
            f"{name} score must be a number, got {type(score).__name__}"
        )
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RewardError(f"{name} score must be between 0 and 1, got {score!r}")
    return value


def all_tests_passed(tests_passed: int, tests_total: int) -> bool:
    """Whether the gate is open: every test passed and there was at least one test."""
    _check_counts(tests_passed, tests_total)
    return tests_total > 0 and tests_passed == tests_total


def code_score(tests_passed: int, tests_total: int) -> float:
    """The fraction of the task tests that passed.

    Zero tests is an error, not a zero score. A rollout with no tests proves
    nothing, and a quiet division would hide a broken runner.
    """
    _check_counts(tests_passed, tests_total)
    if tests_total == 0:
        raise RewardError(
            f"a rollout with zero tests has no code score, got passed={tests_passed} total=0"
        )
    return tests_passed / tests_total


def reward_code_score(
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
) -> float:
    """Code-score arm: the fraction of tests passed, with no judge and no gate.

    The extra score arguments are accepted and ignored, so every arm function
    has the same call shape at the dispatch point.
    """
    return code_score(tests_passed, tests_total)


def reward_hybrid(
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
) -> float:
    """Hybrid arm: zero on any test failure, else `0.5 + 0.5 * judge`."""
    _check_counts(tests_passed, tests_total)
    if not all_tests_passed(tests_passed, tests_total):
        return 0.0
    judge = _check_score("judge", judge_score)
    return 0.5 + 0.5 * judge


def reward_gated_judge(
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
) -> float:
    """Gated judge arm: zero on any test failure, else the judge score."""
    _check_counts(tests_passed, tests_total)
    if not all_tests_passed(tests_passed, tests_total):
        return 0.0
    return _check_score("judge", judge_score)


def reward_gated_process(
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
) -> float:
    """Gated process arm: zero on any test failure, else half trace, half code."""
    _check_counts(tests_passed, tests_total)
    if not all_tests_passed(tests_passed, tests_total):
        return 0.0
    process = _check_score("process", process_score)
    judge = _check_score("judge", judge_score)
    return 0.5 * process + 0.5 * judge


#: The one dispatch point.
ARM_REWARDS: dict[str, RewardFn] = {
    "code-score": reward_code_score,
    "hybrid": reward_hybrid,
    "gated-judge": reward_gated_judge,
    "gated-process": reward_gated_process,
}

#: Arms that are named but have no function yet. Derived from the table, so
#: there is one source of truth.
PENDING_ARMS = frozenset(ARMS) - frozenset(ARM_REWARDS)


def check_arm_implemented(arm: str) -> None:
    """Reject an unknown arm, and an arm that has no reward function yet."""
    _check_arm(arm)
    if arm in PENDING_ARMS:
        raise ArmNotImplemented(
            f"arm {arm!r} is not implemented yet. "
            f"implemented arms: {', '.join(ARM_REWARDS)}."
        )


def compose_reward(
    arm: str,
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
) -> float:
    """The reward for one rollout, dispatched by arm name."""
    check_arm_implemented(arm)
    _check_counts(tests_passed, tests_total)
    return ARM_REWARDS[arm](
        tests_passed=tests_passed,
        tests_total=tests_total,
        judge_score=judge_score,
        process_score=process_score,
    )
