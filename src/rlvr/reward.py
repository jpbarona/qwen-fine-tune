"""Reward composition for the four arms.

The table comes from SPEC.md and is frozen:

| Arm            | Tests fail               | Tests pass                  |
|----------------|--------------------------|-----------------------------|
| code-score     | fraction of tests passed | 1.0                         |
| hybrid         | 0                        | 0.5 + 0.5 * judge           |
| gated-judge    | 0                        | judge                       |
| gated-process  | 0                        | 0.5 * process + 0.5 * judge |

Each arm gets one small function, and `ARM_REWARDS` is the single dispatch
point. Task 4 implements `code-score` only. The other three arms are named in
`ARMS`, but they raise `ArmNotImplemented` before a run starts, until their
tasks add a function to the table. Adding an arm means adding an entry here.
The Experiment loop never changes.

"Tests fail" means any test failed, so a rollout that passes nothing and one
that passes all but one both count as failures for the gated arms. A task with
zero tests is an error for the code score: nothing was proven, and returning
0.0 would hide a broken test runner.
"""

from __future__ import annotations

from typing import Callable

__all__ = [
    "ARM_REWARDS",
    "ARMS",
    "ArmNotImplemented",
    "RewardError",
    "all_tests_passed",
    "check_arm_implemented",
    "code_score",
    "compose_reward",
    "reward_code_score",
    "uses_output_judge",
    "uses_process_judge",
]

#: Arm names in the order used by the CLI and the run folders.
ARMS: tuple[str, ...] = ("code-score", "hybrid", "gated-judge", "gated-process")

#: Arms whose reward is composed from judge output.
OUTPUT_JUDGE_ARMS = frozenset({"hybrid", "gated-judge", "gated-process"})

#: Arms whose reward is composed from the process judge.
PROCESS_JUDGE_ARMS = frozenset({"gated-process"})

#: Arms that have no reward function yet. Remove a name here when its task
#: adds the function to `ARM_REWARDS`.
PENDING_ARMS = frozenset({"hybrid", "gated-judge", "gated-process"})

RewardFn = Callable[..., float]


class RewardError(ValueError):
    """A reward could not be composed, because an arm or a score is wrong."""


class ArmNotImplemented(NotImplementedError):
    """The arm has a name but no reward yet. A run must stop before it starts."""


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise RewardError(f"unknown arm {arm!r}. valid arms: {', '.join(ARMS)}")


def check_arm_implemented(arm: str) -> None:
    """Fail early when an arm has no reward function yet."""
    _check_arm(arm)
    if arm in PENDING_ARMS:
        raise ArmNotImplemented(
            f"arm {arm!r} is not implemented yet, only 'code-score' is. "
            "The other three arms land in Tasks 5 and 6."
        )


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


#: The one dispatch point. Tasks 5 and 6 add their arm functions here.
ARM_REWARDS: dict[str, RewardFn] = {
    "code-score": reward_code_score,
}


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
