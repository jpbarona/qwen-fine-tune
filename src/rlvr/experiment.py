"""The Experiment orchestration seam.

This is the single seam named in SPEC.md. `Experiment` gets four injected
services: a `Generator`, a `Judge`, a `TestRunner`, and a `Store`. Nothing else
in the package talks to a model, an API, Docker, or the filesystem during a run.

Interfaces, frozen for later tasks:

    Generator.generate(prompt, *, n, temperature) -> list[Generation]
    TestRunner.run(prompt, code) -> Outcome
    Judge.score_output(prompt, code) -> JudgeResult
    Judge.score_process(prompt, reasoning) -> JudgeResult
    Store.write_meta(meta)
    Store.append_rollout(record)
    Store.append_metric(row)
    Store.write_eval(payload)
    Store.timestamp, Store.run_dir, Store.arm

`temperature` is 0.0 for greedy evaluation and the sampling temperature during
training. `n` is the number of generations requested for one prompt.

The fake path uses n=1 per prompt. GRPO will pass 8. The signatures stay the
same, so only the loop changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from . import repro
from .config import Config
from .reward import (
    ARMS,
    all_tests_passed,
    code_score,
    compose_reward,
    uses_output_judge,
    uses_process_judge,
)
from .schema import RolloutRecord

__all__ = [
    "Experiment",
    "Generation",
    "Generator",
    "Judge",
    "JudgeResult",
    "Prompt",
    "RunSummary",
    "Store",
    "Outcome",
    "TestRunner",
]

#: Sampling temperature for training generations. Task 4 reads it from config.
TRAINING_TEMPERATURE = 0.9

#: Greedy decoding, used for evaluation.
EVAL_TEMPERATURE = 0.0


@dataclass(frozen=True)
class Prompt:
    """One problem to solve. Task 2 adds the hidden tests to the runner side."""

    prompt_id: str
    problem: str


@dataclass(frozen=True)
class Generation:
    """One model output, with reasoning kept apart from code."""

    reasoning: str
    code: str
    prompt_tokens: int
    completion_tokens: int
    mean_entropy: float


@dataclass(frozen=True)
class Outcome:
    """What happened when the task tests ran against generated code."""

    tests_passed: int
    tests_total: int
    error_class: str


@dataclass(frozen=True)
class JudgeResult:
    """A judge score in [0, 1] and whether the answer came from the cache."""

    score: float
    cache_hit: bool


class Generator(Protocol):
    def generate(self, prompt: Prompt, *, n: int, temperature: float) -> list[Generation]: ...


class TestRunner(Protocol):
    # The name starts with "Test", so tell pytest this is not a test class.
    __test__ = False

    def run(self, prompt: Prompt, code: str) -> Outcome: ...


class Judge(Protocol):
    def score_output(self, prompt: Prompt, code: str) -> JudgeResult: ...

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult: ...


class Store(Protocol):
    arm: str
    timestamp: str
    run_dir: Path

    def write_meta(self, meta: Mapping[str, Any]) -> None: ...

    def append_rollout(self, record: RolloutRecord) -> None: ...

    def append_metric(self, row: Mapping[str, Any]) -> None: ...

    def write_eval(self, payload: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class RunSummary:
    """What one run produced."""

    arm: str
    run_dir: Path
    steps: int
    rollouts: int
    mean_reward: float
    pass_at_1: float


@dataclass
class Experiment:
    """Runs one arm end to end using the injected backends."""

    arm: str
    config: Config
    generator: Generator
    judge: Judge
    test_runner: TestRunner
    store: Store
    prompts: Sequence[Prompt]

    def run(self) -> RunSummary:
        """Run every training step, then evaluate, and return a summary.

        Records are not held in memory. Only running sums are kept, so a long
        run costs the same as a short one.
        """
        self._check_arm()
        if not self.prompts:
            raise ValueError("experiment needs at least one prompt")

        self.store.write_meta(self.build_meta())

        rollouts = 0
        reward_sum = 0.0
        for step in range(1, self.config.steps + 1):
            step_records = [self._one_rollout(prompt, step) for prompt in self.prompts]
            for record in step_records:
                self.store.append_rollout(record)
                rollouts += 1
                reward_sum += record.reward
            self.store.append_metric(self._metric_row(step, step_records))

        evaluation = self._evaluate(self.config.steps)
        self.store.write_eval(self._eval_payload([evaluation]))

        return RunSummary(
            arm=self.arm,
            run_dir=self.store.run_dir,
            steps=self.config.steps,
            rollouts=rollouts,
            mean_reward=reward_sum / rollouts,
            pass_at_1=evaluation["pass_at_1"],
        )

    def build_meta(self) -> dict[str, Any]:
        """The reproducibility record. Written to meta.json."""
        return {
            "arm": self.arm,
            "timestamp": self.store.timestamp,
            "git_commit": repro.git_commit(),
            "versions": repro.package_versions(),
            "model_revision": self.config.model_revision,
            "judge_model": self.config.judge_model,
            "split_seed": self.config.split_seed,
            "config": self.config.raw,
        }

    def _check_arm(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}. valid arms: {', '.join(ARMS)}")

    def _one_rollout(self, prompt: Prompt, step: int) -> RolloutRecord:
        generation = self.generator.generate(
            prompt, n=1, temperature=TRAINING_TEMPERATURE
        )[0]
        result = self.test_runner.run(prompt, generation.code)
        passed = all_tests_passed(result.tests_passed, result.tests_total)

        judge_score: float | None = None
        process_score: float | None = None
        cache_hit = False

        # Gated arms only judge code that passed every test. A failed rollout
        # costs no judge call.
        if passed and uses_output_judge(self.arm):
            judged = self.judge.score_output(prompt, generation.code)
            judge_score = judged.score
            cache_hit = cache_hit or judged.cache_hit
        if passed and uses_process_judge(self.arm):
            judged = self.judge.score_process(prompt, generation.reasoning)
            process_score = judged.score
            cache_hit = cache_hit or judged.cache_hit

        reward = compose_reward(
            self.arm,
            tests_passed=result.tests_passed,
            tests_total=result.tests_total,
            judge_score=judge_score,
            process_score=process_score,
        )

        return RolloutRecord(
            arm=self.arm,
            step=step,
            prompt_id=prompt.prompt_id,
            reasoning=generation.reasoning,
            code=generation.code,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            mean_entropy=generation.mean_entropy,
            tests_passed=result.tests_passed,
            tests_total=result.tests_total,
            error_class=result.error_class,
            code_score=code_score(result.tests_passed, result.tests_total),
            judge_score=judge_score,
            process_score=process_score,
            reward=reward,
            cache_hit=cache_hit,
        )

    @staticmethod
    def _metric_row(step: int, records: Sequence[RolloutRecord]) -> dict[str, Any]:
        count = len(records)
        passed = sum(
            1
            for record in records
            if all_tests_passed(record.tests_passed, record.tests_total)
        )
        return {
            "step": step,
            "arm": records[0].arm if records else None,
            "rollouts": count,
            "mean_reward": sum(record.reward for record in records) / count,
            "pass_rate": passed / count,
            "mean_entropy": sum(record.mean_entropy for record in records) / count,
            "mean_completion_tokens": sum(record.completion_tokens for record in records) / count,
            "cache_hits": sum(1 for record in records if record.cache_hit),
        }

    def _evaluate(self, step: int) -> dict[str, Any]:
        """Greedy pass@1 over the prompts in this run."""
        prompt_ids: list[str] = []
        passed_count = 0
        for prompt in self.prompts:
            generation = self.generator.generate(prompt, n=1, temperature=EVAL_TEMPERATURE)[0]
            result = self.test_runner.run(prompt, generation.code)
            prompt_ids.append(prompt.prompt_id)
            if all_tests_passed(result.tests_passed, result.tests_total):
                passed_count += 1
        total = len(prompt_ids)
        return {
            "step": step,
            "pass_at_1": passed_count / total,
            "num_prompts": total,
            "prompt_ids": prompt_ids,
        }

    def _eval_payload(self, evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        final = evaluations[-1]
        return {
            "arm": self.arm,
            "timestamp": self.store.timestamp,
            "pass_at_1": final["pass_at_1"],
            "num_prompts": final["num_prompts"],
            "eval_prompt_ids": list(final["prompt_ids"]),
            "evaluations": [dict(evaluation) for evaluation in evaluations],
        }
