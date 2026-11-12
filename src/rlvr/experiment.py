"""The Experiment orchestration seam.

This is the single seam named in SPEC.md. `Experiment` gets four injected
services: a `Generator`, a `Judge`, a `TestRunner`, and a `Store`. Nothing else
in the package talks to a model, an API, Docker, or the filesystem during a run.

Interfaces, frozen for later tasks:

    Generator.generate(prompt, *, n, temperature) -> list[Generation]
    Generator.update(prompt, generations, rewards) -> None
    Generator.save_checkpoint(directory) -> None
    Generator.load_checkpoint(directory) -> None
    TestRunner.run(prompt, code) -> Outcome
    Judge.score_output(prompt, code) -> JudgeResult
    Judge.score_process(prompt, reasoning) -> JudgeResult
    Store.write_meta(meta)
    Store.append_rollout(record)
    Store.append_metric(row)
    Store.write_eval(payload)
    Store.timestamp, Store.run_dir, Store.checkpoints_dir

`temperature` is 0.0 for greedy evaluation and the sampling temperature during
training. `n` is the number of generations requested for one prompt.

Training samples one prompt group per step: 8 generations, all tested, judged
only on a full pass per arm, then one `update` call with the group rewards.
Evaluation stays greedy with n=1. Resume and budget guards wrap the same loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from . import repro
from .checkpoint import checkpoint_dir_name, write_checkpoint_manifest
from .config import Config
from .evaluation import pass_at_1, select_eval_subset, subset_selector
from .reward import (
    all_tests_passed,
    check_arm_implemented,
    code_score,
    compose_reward,
    gates_on_tests,
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
    "TestResult",
    "TestRunner",
]

#: Greedy decoding, used for evaluation.
EVAL_TEMPERATURE = 0.0

#: The batch size and gradient accumulation are fixed at 1, so one prompt group
#: is one optimizer step.
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 1


@dataclass(frozen=True)
class Prompt:
    """One problem to solve. Task 2 adds the hidden tests to the runner side."""

    prompt_id: str
    problem: str


@dataclass(frozen=True)
class Generation:
    """One model output, with reasoning kept apart from code.

    `mean_entropy` is the mean Shannon entropy of the model's next-token
    distribution at each generated token position, in nats, averaged over the
    generated tokens only. Prompt tokens are excluded. The value comes from the
    logprobs the generator already receives, so the generator computes it and
    the Experiment only records it. It is logged, not rewarded, and it is not
    bounded by 1. The fake generator returns a deterministic synthetic value.

    `truncated` is true when the completion hit the max length without an end
    token. `prompt_truncated` is true when the prompt was cut at max prompt
    tokens. Neither is part of the rollout record; they feed the step metric.
    """

    reasoning: str
    code: str
    prompt_tokens: int
    completion_tokens: int
    mean_entropy: float
    truncated: bool = False
    prompt_truncated: bool = False


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


class Generator(Protocol):
    def generate(self, prompt: Prompt, *, n: int, temperature: float) -> list[Generation]: ...

    def update(
        self,
        prompt: Prompt,
        generations: Sequence[Generation],
        rewards: Sequence[float],
    ) -> None: ...

    def save_checkpoint(self, directory: Path) -> None: ...

    def load_checkpoint(self, directory: Path) -> None: ...


class TestRunner(Protocol):
    # The name starts with "Test", so tell pytest this is not a test class.
    __test__ = False

    def run(self, prompt: Prompt, code: str) -> Outcome: ...


class Judge(Protocol):
    def score_output(self, prompt: Prompt, code: str) -> JudgeResult: ...

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult: ...

    def score_outputs(
        self, items: Sequence[tuple[Prompt, str]]
    ) -> list[JudgeResult]: ...

    def score_processes(
        self, items: Sequence[tuple[Prompt, str]]
    ) -> list[JudgeResult]: ...


class Store(Protocol):
    arm: str
    timestamp: str
    run_dir: Path
    checkpoints_dir: Path

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
    stop_reason: str = "completed"


@dataclass
class Experiment:
    """Runs one arm end to end using the injected backends.

    `prompts` holds the training questions. `eval_prompts` holds the held-out
    test questions and must be non-empty. The two lists must not overlap. The
    CLI builds both from the fixed MBPP split.
    """

    arm: str
    config: Config
    generator: Generator
    judge: Judge
    test_runner: TestRunner
    store: Store
    prompts: Sequence[Prompt]
    eval_prompts: Sequence[Prompt] = ()
    resume_step: int = 0
    prior_evaluations: Sequence[Mapping[str, Any]] = ()

    def run(self) -> RunSummary:
        """Run every training step, evaluate on schedule, and summarize.

        Records are not held in memory. Only running sums are kept, so a long
        run costs the same as a short one. `eval.json` is replaced after every
        evaluation, so a crash keeps the last checkpoint result.

        Budget guards wrap the loop. `max_steps` caps the step count below
        `config.steps`. `max_judge_calls` stops the loop at the next step
        boundary once the judge has been called that many times. A resume starts
        at `resume_step + 1` and carries forward the evaluations already on
        disk, so records are appended, never duplicated.
        """
        self._check_arm()
        if not self.prompts:
            raise ValueError("experiment needs at least one prompt")
        if not self.eval_prompts:
            raise ValueError(
                "experiment needs at least one evaluation prompt. "
                "Pass the held-out test questions as eval_prompts."
            )

        resumed = self.resume_step > 0
        if not resumed:
            self.store.write_meta(self.build_meta())

        eval_pool = tuple(self.eval_prompts)
        checkpoint_prompts = select_eval_subset(
            eval_pool, count=self.config.eval_questions, seed=self.config.eval_subset_seed
        )
        checkpoint_subset = subset_selector(
            eval_pool,
            count=self.config.eval_questions,
            seed=self.config.eval_subset_seed,
            subset=checkpoint_prompts,
        )

        effective_steps = self.config.steps
        if self.config.max_steps is not None:
            effective_steps = min(effective_steps, self.config.max_steps)

        rollouts = 0
        reward_sum = 0.0
        judge_calls = 0
        last_step = self.resume_step
        stop_reason = "completed"
        evaluations: list[dict[str, Any]] = [dict(e) for e in self.prior_evaluations]

        for step in range(self.resume_step + 1, effective_steps + 1):
            step_records, truncated, prompt_truncated = self._training_step(step)
            last_step = step
            for record in step_records:
                self.store.append_rollout(record)
                rollouts += 1
                reward_sum += record.reward
                judge_calls += self._judge_calls_for(record)
            self.store.append_metric(
                self._metric_row(step, step_records, truncated, prompt_truncated)
            )

            if step % self.config.eval_every_steps == 0:
                evaluations.append(
                    self._evaluate(
                        step,
                        checkpoint_prompts,
                        kind="checkpoint",
                        subset=checkpoint_subset,
                    )
                )
                self.store.write_eval(
                    self._eval_payload(evaluations, checkpoint_subset, stop_reason)
                )

            if step % self.config.checkpoint_every_steps == 0:
                self._save_checkpoint(step)

            if (
                self.config.max_judge_calls is not None
                and judge_calls >= self.config.max_judge_calls
            ):
                stop_reason = "max_judge_calls"
                break

        if stop_reason == "completed" and self.config.max_steps is not None:
            if last_step >= self.config.max_steps and self.config.max_steps < self.config.steps:
                stop_reason = "max_steps"

        final = self._evaluate(last_step, eval_pool, kind="final", subset=None)
        evaluations.append(final)
        self.store.write_eval(
            self._eval_payload(evaluations, checkpoint_subset, stop_reason)
        )

        return RunSummary(
            arm=self.arm,
            run_dir=self.store.run_dir,
            steps=last_step,
            rollouts=rollouts,
            mean_reward=reward_sum / rollouts if rollouts else 0.0,
            pass_at_1=final["pass_at_1"],
            stop_reason=stop_reason,
        )

    def build_meta(self) -> dict[str, Any]:
        """The reproducibility record. Written to meta.json."""
        config = self.config
        return {
            "arm": self.arm,
            "timestamp": self.store.timestamp,
            "git_commit": repro.git_commit(),
            "versions": repro.package_versions(),
            "model_revision": config.model_revision,
            "judge_model": config.judge_model,
            "split_seed": config.split_seed,
            "config": config.raw,
            "trainer": {
                "method": "grpo",
                "model_name": config.model_name,
                "generations_per_prompt": config.generations_per_prompt,
                "epochs": config.epochs,
                "temperature": config.temperature,
                "learning_rate": config.learning_rate,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "max_prompt_tokens": config.max_prompt_tokens,
                "max_completion_tokens": config.max_completion_tokens,
                "clip_epsilon": config.clip_epsilon,
                "kl_beta": config.kl_beta,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
            },
            "checkpoint_dir": str(self.store.checkpoints_dir),
        }

    def _check_arm(self) -> None:
        """Reject an unknown arm, and an arm that has no reward yet.

        Called before `meta.json` is written, so a pending arm leaves no run
        folder and touches no backend.
        """
        check_arm_implemented(self.arm)

    def _prompt_for_step(self, step: int) -> Prompt:
        """The one prompt this step trains on. Batch size is 1, so one prompt
        group per step, cycling through the training prompts."""
        return self.prompts[(step - 1) % len(self.prompts)]

    def _training_step(self, step: int) -> tuple[list[RolloutRecord], int, int]:
        """Sample one prompt group, run and judge all of it, update once.

        All 8 generations are tested. The passing ones are judged in two batch
        calls, one for output scores and one for process scores, so the 32-way
        judge concurrency is used. A failed rollout still costs no judge call.
        The group's rewards are passed to `update` once. Returns the 8 records
        plus the counts of truncated and prompt-truncated generations.
        """
        prompt = self._prompt_for_step(step)
        generations = self.generator.generate(
            prompt,
            n=self.config.generations_per_prompt,
            temperature=self.config.temperature,
        )
        outcomes = [
            self.test_runner.run(prompt, generation.code) for generation in generations
        ]
        passing = [
            index
            for index, outcome in enumerate(outcomes)
            if all_tests_passed(outcome.tests_passed, outcome.tests_total)
        ]

        judge_scores: list[float | None] = [None] * len(generations)
        process_scores: list[float | None] = [None] * len(generations)
        cache_hits = [False] * len(generations)

        if passing and uses_output_judge(self.arm):
            items = [(prompt, generations[index].code) for index in passing]
            for index, result in zip(passing, self.judge.score_outputs(items)):
                judge_scores[index] = result.score
                cache_hits[index] = result.cache_hit
        if passing and uses_process_judge(self.arm):
            items = [(prompt, generations[index].reasoning) for index in passing]
            for index, result in zip(passing, self.judge.score_processes(items)):
                process_scores[index] = result.score
                cache_hits[index] = cache_hits[index] or result.cache_hit

        records = [
            self._build_record(
                prompt,
                generation,
                outcome,
                step,
                judge_score=judge_scores[index],
                process_score=process_scores[index],
                cache_hit=cache_hits[index],
            )
            for index, (generation, outcome) in enumerate(zip(generations, outcomes))
        ]

        rewards = [record.reward for record in records]
        self.generator.update(prompt, generations, rewards)
        truncated = sum(1 for generation in generations if generation.truncated)
        prompt_truncated = sum(
            1 for generation in generations if generation.prompt_truncated
        )
        return records, truncated, prompt_truncated

    def _build_record(
        self,
        prompt: Prompt,
        generation: Generation,
        outcome: Outcome,
        step: int,
        *,
        judge_score: float | None,
        process_score: float | None,
        cache_hit: bool,
    ) -> RolloutRecord:
        reward = compose_reward(
            self.arm,
            tests_passed=outcome.tests_passed,
            tests_total=outcome.tests_total,
            judge_score=judge_score,
            process_score=process_score,
        )

        # A question with no tests cannot open the gate. The gated arms record a
        # code score of 0.0 and reward 0.0, with no judge call. The code-score
        # arm still fails loudly: there, zero tests is a broken test runner.
        if outcome.tests_total == 0 and gates_on_tests(self.arm):
            record_code_score = 0.0
        else:
            record_code_score = code_score(outcome.tests_passed, outcome.tests_total)

        return RolloutRecord(
            arm=self.arm,
            step=step,
            prompt_id=prompt.prompt_id,
            reasoning=generation.reasoning,
            code=generation.code,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            mean_entropy=generation.mean_entropy,
            tests_passed=outcome.tests_passed,
            tests_total=outcome.tests_total,
            error_class=outcome.error_class,
            code_score=record_code_score,
            judge_score=judge_score,
            process_score=process_score,
            reward=reward,
            cache_hit=cache_hit,
        )

    @staticmethod
    def _judge_calls_for(record: RolloutRecord) -> int:
        """The number of judge calls a record represents.

        `judge_score` is set exactly when the output judge was called and
        `process_score` exactly when the process judge was called, so the two
        null checks count calls precisely.
        """
        calls = 0
        if record.judge_score is not None:
            calls += 1
        if record.process_score is not None:
            calls += 1
        return calls

    def _save_checkpoint(self, step: int) -> None:
        """Ask the generator for its state, then write the checkpoint manifest."""
        directory = self.store.checkpoints_dir / checkpoint_dir_name(step)
        self.generator.save_checkpoint(directory)
        write_checkpoint_manifest(
            directory, step=step, arm=self.arm, timestamp=self.store.timestamp
        )

    @staticmethod
    def _metric_row(
        step: int,
        records: Sequence[RolloutRecord],
        truncated: int = 0,
        prompt_truncated: int = 0,
    ) -> dict[str, Any]:
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
            "truncation_rate": truncated / count,
            "prompt_truncation_rate": prompt_truncated / count,
        }

    def _evaluate(
        self,
        step: int,
        prompts: Sequence[Prompt],
        *,
        kind: str,
        subset: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Greedy pass@1 over `prompts`. One generation per question."""
        prompt_ids: list[str] = []
        passed: list[bool] = []
        for prompt in prompts:
            generation = self.generator.generate(prompt, n=1, temperature=EVAL_TEMPERATURE)[0]
            result = self.test_runner.run(prompt, generation.code)
            prompt_ids.append(prompt.prompt_id)
            passed.append(all_tests_passed(result.tests_passed, result.tests_total))
        return {
            "step": step,
            "kind": kind,
            "pass_at_1": pass_at_1(passed),
            "count": len(prompt_ids),
            "prompt_ids": prompt_ids,
            "subset": dict(subset) if subset is not None else None,
        }

    def _eval_payload(
        self,
        evaluations: Sequence[Mapping[str, Any]],
        checkpoint_subset: Mapping[str, Any],
        stop_reason: str,
    ) -> dict[str, Any]:
        """The whole eval.json. The last evaluation is the final one."""
        final = evaluations[-1]
        return {
            "arm": self.arm,
            "timestamp": self.store.timestamp,
            "step": final["step"],
            "pass_at_1": final["pass_at_1"],
            "count": final["count"],
            "subset": dict(checkpoint_subset),
            "stop_reason": stop_reason,
            "evaluations": [dict(evaluation) for evaluation in evaluations],
        }
