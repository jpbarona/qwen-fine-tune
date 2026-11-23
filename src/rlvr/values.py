"""The values that cross the experiment seam.

The training loop, the generator, the test runner, the judge and the store
all exchange these small frozen objects. They live here, not in the training
loop, so a backend can name a prompt without importing the loop that drives
it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """One problem to solve. Task 2 adds the hidden tests to the runner side."""

    prompt_id: str
    problem: str


@dataclass(frozen=True)
class Generation:
    """One model output, with reasoning kept apart from code.

    Entropy is not part of a generation. The update pass measures it at the
    configured cadence, on the logits it already computes, and the Experiment
    records it on the step metric row. It is logged, not rewarded.

    `truncated` is true when the completion hit the max length without an end
    token. `prompt_truncated` is true when the prompt was cut at max prompt
    tokens. Neither is part of the rollout record; they feed the step metric.

    `closed` is true when the completion wrote the closing reasoning marker, so
    the reasoning block ended before the answer instead of running into the cap.
    The code arm pays a small bonus for it so the model has a reason to close.
    """

    reasoning: str
    code: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False
    prompt_truncated: bool = False
    closed: bool = False


@dataclass(frozen=True)
class TestResult:
    """One task test, run on its own. `error_class` is `"pass"` when it passed."""

    # The name starts with "Test", so tell pytest this is not a test class.
    __test__ = False

    index: int
    passed: bool
    error_class: str = "pass"
    detail: str = ""


@dataclass(frozen=True)
class Outcome:
    """What happened when the task tests ran against generated code.

    `results` holds one entry per task test. It defaults to empty so a runner
    that only counts passes stays valid. The rollout record keeps the counts
    and the overall class, as frozen in Task 1.
    """

    tests_passed: int
    tests_total: int
    error_class: str
    results: tuple[TestResult, ...] = ()


@dataclass(frozen=True)
class JudgeResult:
    """A judge score in [0, 1], whether the answer came from the cache, and
    the short reason the judge returned.

    `reason` has a default so every existing fake and test stays valid. The
    score and cache fields are the frozen part of this seam.
    """

    score: float
    cache_hit: bool
    reason: str = ""
