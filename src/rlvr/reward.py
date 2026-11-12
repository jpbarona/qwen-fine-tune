"""Reward composition for the four arms.

The table comes from SPEC.md and is frozen:

| Arm            | Tests fail              | Tests pass                  |
|----------------|-------------------------|-----------------------------|
| code-score     | fraction of tests passed| 1.0                         |
| hybrid         | 0                       | 0.5 + 0.5 * judge           |
| gated-judge    | 0                       | judge                       |
| gated-process  | 0                       | 0.5 * process + 0.5 * judge |

"Tests fail" means any test failed, so a rollout that passes nothing and one that
passes all but one both count as failures for the gated arms. A task with zero
tests is a failure: nothing was proven.
"""

from __future__ import annotations

__all__ = ["ARMS", "RewardError", "code_score", "compose_reward",
           "uses_output_judge", "uses_process_judge"]

#: Arm names in the order used by the CLI and the run folders.
ARMS: tuple[str, ...] = ("code-score", "hybrid", "gated-judge", "gated-process")

OUTPUT_JUDGE_ARMS = frozenset({"hybrid", "gated-judge", "gated-process"})
PROCESS_JUDGE_ARMS = frozenset({"gated-process"})


class RewardError(ValueError):
    """A reward could not be composed, because an arm or a score is wrong."""


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


def _check_score(name: str, score: float | None) -> float:
    if score is None:
        raise RewardError(f"this arm needs a {name} score, got null")
    if not 0.0 <= float(score) <= 1.0:
        raise RewardError(f"{name} score must be between 0 and 1, got {score!r}")
    return float(score)


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
    """The fraction of the task tests that passed. Zero tests gives 0.0."""
    _check_counts(tests_passed, tests_total)
    if tests_total == 0:
        return 0.0
    return tests_passed / tests_total


def compose_reward(
    arm: str,
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
) -> float:
    """The reward for one rollout, following the arm table above."""
    _check_arm(arm)
    _check_counts(tests_passed, tests_total)

    if arm == "code-score":
        return code_score(tests_passed, tests_total)

    if not all_tests_passed(tests_passed, tests_total):
        return 0.0

    judge = _check_score("judge", judge_score)
    if arm == "hybrid":
        return 0.5 + 0.5 * judge
    if arm == "gated-judge":
        return judge
    process = _check_score("process", process_score)
    return 0.5 * process + 0.5 * judge
