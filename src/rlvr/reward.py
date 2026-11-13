"""Reward composition for the four arms.

The table comes from SPEC.md and is frozen:

| Arm            | Tests fail               | Tests pass                  |
|----------------|--------------------------|-----------------------------|
| code-score     | see below                | 1.0                         |
| hybrid         | 0                        | 0.5 + 0.5 * judge           |
| gated-judge    | 0                        | judge                       |
| gated-process  | 0                        | 0.5 * process + 0.5 * judge |

The code arm now gives partial credit and penalties. Its base score is the
fraction of tests passed, except that a truncated rollout scores -0.6 and a
rollout whose code does not compile scores -0.5. A length term of 0.001 per
completion token above 900 is subtracted from that base, capped at 0.2, so a
perfect answer keeps at least 0.8. The sum is clipped to -1 to 1. The other
three arms are unchanged.

Each arm gets one small function, and `ARM_REWARDS` is the single dispatch
point. All four arms are implemented. Adding an arm means adding one entry to
`ARM_REWARDS`. The Experiment loop never changes.

"Tests fail" means any test failed, so a rollout that passes nothing and one
that passes all but one both count as failures for the gated arms. A task with
zero tests is an error for the code score: nothing was proven, and returning
0.0 would hide a broken test runner. The one exception is a syntax error: the
code never ran, so the runner knows no test counts, and the arm scores -0.5.
The gated arms differ: a question with no tests cannot pass the gate, so it gets
zero and the judge is not called. The Experiment records a code score of 0.0 for
that case.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = [
    "ARM_REWARDS",
    "ARMS",
    "ArmNotImplemented",
    "DEFAULT_LENGTH_PENALTY_FLOOR",
    "DEFAULT_LENGTH_PENALTY_PER_TOKEN",
    "DEFAULT_LENGTH_SOFT_LIMIT",
    "PENDING_ARMS",
    "RewardError",
    "SYNTAX_ERROR_CLASS",
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

#: Defaults for the code-arm length term. The config loader uses the same names.
DEFAULT_LENGTH_SOFT_LIMIT = 900
DEFAULT_LENGTH_PENALTY_PER_TOKEN = 0.001
DEFAULT_LENGTH_PENALTY_FLOOR = 0.2

#: Fixed code-arm scores for outcomes that are not a test fraction.
TRUNCATION_REWARD = -0.6
SYNTAX_ERROR_REWARD = -0.5

#: The code arm clips its sum to this range.
MIN_REWARD = -1.0
MAX_REWARD = 1.0

#: The error class for code that does not compile. A rollout with no extracted
#: code is reported this way by the runner, so there is no separate class.
SYNTAX_ERROR_CLASS = "syntax_error"

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
    code-score arm has no gate: its base score is the fraction of tests passed.
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


def _length_penalty(
    completion_tokens: int | None,
    *,
    soft_limit: int,
    per_token: float,
    floor: float,
) -> float:
    """The length term: `per_token` for each token above `soft_limit`, capped at `floor`.

    `floor` is a floor on the reward, not on the term. The term never removes
    more than `floor`, so a perfect answer keeps at least `1 - floor`.
    """
    if completion_tokens is None or completion_tokens <= soft_limit:
        return 0.0
    return min(floor, per_token * (completion_tokens - soft_limit))


def reward_code_score(
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
    truncated: bool = False,
    error_class: str | None = None,
    completion_tokens: int | None = None,
    length_soft_limit: int = DEFAULT_LENGTH_SOFT_LIMIT,
    length_penalty_per_token: float = DEFAULT_LENGTH_PENALTY_PER_TOKEN,
    length_penalty_floor: float = DEFAULT_LENGTH_PENALTY_FLOOR,
) -> float:
    """Code-score arm: the test fraction, with truncation, syntax and length terms.

    A truncated rollout scores -0.6 and a rollout whose code did not compile
    scores -0.5. Everything else starts at the fraction of tests passed. The
    length term is subtracted from that base and the sum is clipped to [-1, 1].

    The judge score arguments are accepted and ignored, so every arm function
    has the same call shape at the dispatch point.
    """
    _check_counts(tests_passed, tests_total)
    if truncated:
        base = TRUNCATION_REWARD
    elif error_class == SYNTAX_ERROR_CLASS:
        base = SYNTAX_ERROR_REWARD
    else:
        base = code_score(tests_passed, tests_total)
    penalty = _length_penalty(
        completion_tokens,
        soft_limit=length_soft_limit,
        per_token=length_penalty_per_token,
        floor=length_penalty_floor,
    )
    return min(MAX_REWARD, max(MIN_REWARD, base - penalty))


def reward_hybrid(
    *,
    tests_passed: int,
    tests_total: int,
    judge_score: float | None = None,
    process_score: float | None = None,
    truncated: bool = False,
    error_class: str | None = None,
    completion_tokens: int | None = None,
    length_soft_limit: int = DEFAULT_LENGTH_SOFT_LIMIT,
    length_penalty_per_token: float = DEFAULT_LENGTH_PENALTY_PER_TOKEN,
    length_penalty_floor: float = DEFAULT_LENGTH_PENALTY_FLOOR,
) -> float:
    """Hybrid arm: zero on any test failure, else `0.5 + 0.5 * judge`.

    The rollout context and length settings are accepted and ignored.
    """
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
    truncated: bool = False,
    error_class: str | None = None,
    completion_tokens: int | None = None,
    length_soft_limit: int = DEFAULT_LENGTH_SOFT_LIMIT,
    length_penalty_per_token: float = DEFAULT_LENGTH_PENALTY_PER_TOKEN,
    length_penalty_floor: float = DEFAULT_LENGTH_PENALTY_FLOOR,
) -> float:
    """Gated judge arm: zero on any test failure, else the judge score.

    The rollout context and length settings are accepted and ignored.
    """
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
    truncated: bool = False,
    error_class: str | None = None,
    completion_tokens: int | None = None,
    length_soft_limit: int = DEFAULT_LENGTH_SOFT_LIMIT,
    length_penalty_per_token: float = DEFAULT_LENGTH_PENALTY_PER_TOKEN,
    length_penalty_floor: float = DEFAULT_LENGTH_PENALTY_FLOOR,
) -> float:
    """Gated process arm: zero on any test failure, else half trace, half code.

    The rollout context and length settings are accepted and ignored.
    """
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
    truncated: bool = False,
    error_class: str | None = None,
    completion_tokens: int | None = None,
    length_soft_limit: int = DEFAULT_LENGTH_SOFT_LIMIT,
    length_penalty_per_token: float = DEFAULT_LENGTH_PENALTY_PER_TOKEN,
    length_penalty_floor: float = DEFAULT_LENGTH_PENALTY_FLOOR,
) -> float:
    """The reward for one rollout, dispatched by arm name.

    The rollout context and the length settings are passed to every arm. Only
    the code arm uses them. The others accept the same shape and ignore it.
    """
    check_arm_implemented(arm)
    _check_counts(tests_passed, tests_total)
    return ARM_REWARDS[arm](
        tests_passed=tests_passed,
        tests_total=tests_total,
        judge_score=judge_score,
        process_score=process_score,
        truncated=truncated,
        error_class=error_class,
        completion_tokens=completion_tokens,
        length_soft_limit=length_soft_limit,
        length_penalty_per_token=length_penalty_per_token,
        length_penalty_floor=length_penalty_floor,
    )
