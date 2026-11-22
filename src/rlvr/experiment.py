"""The Experiment orchestration seam.

This is the single seam named in SPEC.md. `Experiment` gets four injected
services: a `Generator`, a `Judge`, a `TestRunner`, and a `Store`. Nothing else
in the package talks to a model, an API, Docker, or the filesystem during a run.

Interfaces, frozen for later tasks:

    Generator.generate(prompt, *, n, temperature) -> list[Generation]
    Generator.generate_batch(prompts, *, n, temperature) -> list[list[Generation]]
    Generator.accumulate(prompt, generations, rewards, *, measure_entropy) -> float | None
    Generator.apply_update() -> float | None
    Generator.save_checkpoint(directory) -> None
    Generator.load_checkpoint(directory) -> None
    TestRunner.run(prompt, code) -> Outcome
    Judge.score_output(prompt, code) -> JudgeResult
    Judge.score_process(prompt, reasoning) -> JudgeResult
    Store.write_meta(meta)
    Store.append_rollout(record)
    Store.append_metric(row)
    Store.append_label(row)
    Store.read_labels()
    Store.write_eval(payload)
    Store.timestamp, Store.run_dir, Store.checkpoints_dir

`temperature` is 0.0 for greedy evaluation and the sampling temperature during
training. `n` is the number of generations requested for one prompt.

Training samples several prompt groups per step, in one `generate_batch` call.
Each prompt keeps its own group of generations, and each group is tested,
judged only on a full pass per arm, then has its gradient accumulated on its own
rewards. Advantage is normalised inside one prompt group, never across the step.
The optimizer steps once at the end of the step, so the step counter is also the
number of weight updates. Without a schedule the prompt count starts at two and
grows while free memory stays above the floor. With a schedule the rollout
budget sets the prompt count, so the two policies compared later get equal
compute.

A run trains from one measured pool. The pool is built from pass counts before
the run. The step draw asks a `Schedule` for a `StepPlan` of train prompts and
parked prompts. Parked rollouts join the same generator call and the same
optimizer update, so their groups can still produce a gradient. A schedule
without a policy trains the train pool alone. There is no curriculum and no
stage ladder. Every record carries the fixed `unstaged` stage value so older
readers still find the field. Evaluation stays greedy with n=1.

Pool label changes are appended to `labels.jsonl` in the run folder. A resume
reads that file before the first step, so a promoted or demoted prompt keeps its
pool. A scheduled run also writes its saved queues in that file, so a resume
continues the same rotation. Resume and budget guards wrap the same loop. The
step counter counts batches and weight updates, so a resume starts at the batch
after the checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import repro
from .control import ControlState, ControlWatcher
from .batching import START_PROMPTS, BatchSizePolicy, is_cuda_out_of_memory
from .checkpoint import (
    checkpoint_dir_name,
    parse_checkpoint_step,
    prune_checkpoints,
    select_checkpoints_to_keep,
    write_checkpoint_manifest,
)
from .config import Config
from .evaluation import pass_at_1, select_eval_subset, subset_selector
from .memory import MemoryReport
from .pools import DEFAULT_PROMOTION_BAR, Pools
from .queues import queues_from_row, restore_queue
from .schedule import (
    GroupOutcome,
    LabelUpdate,
    Schedule,
    Spread,
    StepPlan,
    Sweep,
    label_from_row,
    streak_from_row,
    sweep_order_from_row,
    sweep_position_from_row,
    sweep_steps_from_row,
)
from .reward import (
    SYNTAX_ERROR_CLASS,
    all_tests_passed,
    check_arm_implemented,
    code_score,
    compose_reward,
    gates_on_tests,
    uses_output_judge,
    uses_process_judge,
)
from .schema import RolloutRecord
from .stages import UNSTAGED_STAGE

__all__ = [
    "Experiment",
    "Generation",
    "Generator",
    "Judge",
    "JudgeResult",
    "Prompt",
    "RunSummary",
    "Schedule",
    "StepPlan",
    "Store",
    "Outcome",
    "TestResult",
    "TestRunner",
]

#: Greedy decoding, used for evaluation.
EVAL_TEMPERATURE = 0.0


def _finite_or_none(value: float | None) -> float | None:
    """A metric value the JSON writer accepts, or None when it is not finite."""
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _group_spread_and_sd(
    rewards: Sequence[float | None],
) -> tuple[float, float] | None:
    """The mean absolute deviation and population SD of one prompt group.

    A group of one reward gives zero for both. A group with a missing or
    non-finite reward is skipped by returning None, so the step mean is taken
    over the groups that hold a real reward only.
    """
    values: list[float] = []
    for reward in rewards:
        if reward is None:
            return None
        value = float(reward)
        if not math.isfinite(value):
            return None
        values.append(value)
    if not values:
        return None
    mean = sum(values) / len(values)
    spread = sum(abs(value - mean) for value in values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if not math.isfinite(spread) or not math.isfinite(sd):
        return None
    return spread, sd


def _mean_group_signal(
    groups: Sequence[Sequence[float | None]],
) -> tuple[float, float]:
    """The step mean of the per-group spread and SD.

    Groups that cannot produce a signal are skipped, not counted as zero. A
    step with no usable group reports zero for both fields.
    """
    spreads: list[float] = []
    sds: list[float] = []
    for rewards in groups:
        signal = _group_spread_and_sd(rewards)
        if signal is None:
            continue
        spreads.append(signal[0])
        sds.append(signal[1])
    return (
        sum(spreads) / len(spreads) if spreads else 0.0,
        sum(sds) / len(sds) if sds else 0.0,
    )


def _empty_cuda_cache() -> None:
    """Release cached allocator blocks after an out-of-memory error.

    The CPU suite and the fake path have no torch, so the import failure is the
    normal case there. A failed cache release must not hide the original error.
    """
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (RuntimeError, AssertionError):
        return


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


class Generator(Protocol):
    def generate(self, prompt: Prompt, *, n: int, temperature: float) -> list[Generation]: ...

    def generate_batch(
        self,
        prompts: Sequence[Prompt],
        *,
        n: int,
        temperature: float,
        thinking_budget: int | None = None,
    ) -> list[list[Generation]]: ...

    def accumulate(
        self,
        prompt: Prompt,
        generations: Sequence[Generation],
        rewards: Sequence[float],
        *,
        measure_entropy: bool = False,
    ) -> float | None: ...

    def apply_update(self) -> float | None: ...

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

    def append_label(self, row: Mapping[str, Any]) -> None: ...

    def read_labels(self) -> list[dict[str, Any]]: ...

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
    stage: str = UNSTAGED_STAGE


@dataclass(frozen=True)
class StepResult:
    """What one batched training step produced.

    `prompts` is the number of prompt groups. `train_prompts` and
    `parked_prompts` split that count by the plan. `advantage_share` is the
    number of groups whose rewards hold more than one distinct value, divided
    by the group count. `group_spread` is the mean absolute deviation of each
    group's rewards from its mean, then averaged over the groups.
    `group_sd` is the population standard deviation, then averaged the same
    way. `mastered` lists the prompts that had at least one full test pass. The
    sequence count is the length of `records`.
    """

    records: tuple[RolloutRecord, ...]
    prompts: int
    truncated: int
    prompt_truncated: int
    mean_entropy: float | None
    grad_norm: float | None
    update_norm: float | None
    mastered: tuple[Prompt, ...]
    timings: Mapping[str, float] = field(default_factory=dict)
    train_prompts: int = 0
    parked_prompts: int = 0
    advantage_share: float = 0.0
    group_spread: float = 0.0
    group_sd: float = 0.0
    outcomes: tuple[GroupOutcome, ...] = ()


@dataclass(frozen=True)
class _ScoredGroup:
    records: tuple[RolloutRecord, ...]
    solved: bool
    seconds: float


@dataclass
class Experiment:
    """Runs one arm end to end using the injected backends.

    `prompts` holds the training questions. `eval_prompts` holds the held-out
    test questions and must be non-empty. The two lists must not overlap. The
    CLI builds both from the fixed MBPP split.

    `pools` is the single pool built from measured pass counts. When it is
    present, only ids in `pools.train` and `pools.parked` are held by the
    experiment. The train ids fill the step and the parked ids feed the
    schedule. Retired prompts are kept in the pool record and never trained.
    When it is None, every prompt in `prompts` is trained, which keeps unit
    tests and pre-pool run folders working.

    `schedule` is the step-draw policy. A caller may inject one, which is what
    the unit tests do. When it is None and `config.pool_policy` is set, the
    experiment builds the named policy. When neither is set, the step cycles
    through the train prompts with no parked prompts, which is the placeholder
    path.

    `labels_path` is the difficulty label file used for reporting only. The
    training path never reads it, so a missing file does not stop a run.

    `memory` is an optional callable that reads the allocator for the current
    step. The CLI passes `read_torch_memory` on a real run and `fake_memory` on
    the fake path. When it is None, every metric row carries zeros, so a unit
    test can build an Experiment without a GPU.

    `resume_prompt_cursor` is the position in `prompts` where the first resumed
    step starts. The checkpoint manifest records it, and the CLI passes it back.
    Checkpoint stage fields from older runs are ignored, because the run has one
    pool and no stage position.

    `batch_policy` is injectable so a test can pin the prompt count. A run
    without one builds a policy from the configured memory floor.
    """

    arm: str
    config: Config
    generator: Generator
    judge: Judge
    test_runner: TestRunner
    store: Store
    prompts: Sequence[Prompt]
    pools: Pools | None = None
    labels_path: Path | None = None
    control: ControlWatcher | None = None
    eval_prompts: Sequence[Prompt] = ()
    resume_step: int = 0
    prior_evaluations: Sequence[Mapping[str, Any]] = ()
    fake: bool = False
    memory: Callable[[], MemoryReport] | None = None
    resume_prompt_cursor: int = 0
    batch_policy: BatchSizePolicy | None = None
    schedule: Schedule | None = None
    _prompt_cursor: int = field(init=False, default=0, repr=False)
    _policy: BatchSizePolicy | None = field(init=False, default=None, repr=False)
    _schedule: Schedule | None = field(init=False, default=None, repr=False)
    _parked_prompts: list[Prompt] = field(init=False, default_factory=list, repr=False)
    _saved_streaks: dict[str, int] = field(
        init=False, default_factory=dict, repr=False
    )

    def run(self) -> RunSummary:
        """Run every training step, evaluate on schedule, and summarize.

        Records are not held in memory. Only running sums are kept, so a long
        run costs the same as a short one. `eval.json` is replaced after every
        evaluation, so a crash keeps the last checkpoint result.

        Each step takes the prompt count the batch policy allows and applies one
        optimizer update. The step counter counts weight updates, never prompts.

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

        self.prompts, self._parked_prompts = self._pool_prompts()
        self._schedule = self.schedule or self._build_schedule()
        if self._schedule is not None:
            self._load_labels()
        if not self.prompts:
            raise ValueError(
                "the training pool holds no prompts. Every prompt is parked or "
                "outside the pool."
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

        # A rerun starts from the same place, so a reused Experiment is
        # deterministic. An injected policy is left as the caller set it.
        policy = self.batch_policy or BatchSizePolicy(
            floor_mib=self.config.memory_floor_mib,
            max_prompts=self.config.max_prompts,
        )
        self._policy = policy
        self._prompt_cursor = self.resume_prompt_cursor

        rollouts = 0
        reward_sum = 0.0
        judge_calls = 0
        last_step = self.resume_step
        stop_reason = "completed"
        evaluations: list[dict[str, Any]] = [dict(e) for e in self.prior_evaluations]

        for step in range(self.resume_step + 1, effective_steps + 1):
            if self.control is not None:
                state = self.control.poll()
                self._record_control(step, state)
                for prompt_id, tests in state.tests.items():
                    replace = getattr(self.test_runner, "replace_tests", None)
                    if replace is not None:
                        replace(prompt_id, tests)
                if state.drop:
                    remaining = [
                        prompt
                        for prompt in self.prompts
                        if prompt.prompt_id not in state.drop
                    ]
                    if remaining:
                        self.prompts = remaining
                        self._parked_prompts = [
                            prompt
                            for prompt in self._parked_prompts
                            if prompt.prompt_id not in state.drop
                        ]
                    else:
                        break
            result = self._train_with_memory_guard(
                step,
                policy,
                thinking_budget=self.config.thinking_budget,
            )
            if result is None:
                # The step was attempted and stopped on memory. Record its
                # number so the evaluation payload keeps a positive step.
                last_step = step
                stop_reason = "out_of_memory"
                break
            last_step = step
            for record in result.records:
                self.store.append_rollout(record)
                rollouts += 1
                reward_sum += record.reward
                judge_calls += self._judge_calls_for(record)

            memory = self._memory_report()
            self.store.append_metric(
                self._metric_row(
                    step,
                    result.records,
                    prompts=result.prompts,
                    truncated=result.truncated,
                    prompt_truncated=result.prompt_truncated,
                    mean_entropy=result.mean_entropy,
                    grad_norm=result.grad_norm,
                    update_norm=result.update_norm,
                    memory=memory,
                    timings=result.timings,
                    max_prompts=policy.max_prompts,
                    pool_policy=self._pool_policy_name(),
                    train_prompts=result.train_prompts,
                    parked_prompts=result.parked_prompts,
                    advantage_share=result.advantage_share,
                    group_spread=result.group_spread,
                    group_sd=result.group_sd,
                )
            )
            policy.observe(memory, prompts=result.prompts)
            # Label changes land before the checkpoint, so a resume from this
            # step sees the same pools as the run did.
            self._observe_step(step, result)
            self._record_schedule_state(step)

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
            stage=UNSTAGED_STAGE,
        )

    def build_meta(self) -> dict[str, Any]:
        """The reproducibility record. Written to meta.json."""
        config = self.config
        # A caller-supplied policy may pin a different cap than the config.
        # Record the value the run uses, so meta and the metrics agree.
        policy = self._policy or self.batch_policy
        max_prompts = policy.max_prompts if policy is not None else config.max_prompts
        return {
            "arm": self.arm,
            "timestamp": self.store.timestamp,
            "mode": "fake" if self.fake else "real",
            "git_commit": repro.git_commit(),
            "versions": repro.package_versions(),
            "model_revision": config.model_revision,
            "judge_model": config.judge_model,
            "split_seed": config.split_seed,
            "config": config.raw,
            "pools": self._pools_meta(),
            "pool_policy": self._pool_policy_name(),
            "schedule": self._schedule_meta(),
            "screen_rows": config.screen_rows,
            "labels_path": str(self.labels_path) if self.labels_path is not None else None,
            "trainer": {
                "method": "grpo",
                "model_name": config.model_name,
                "generations_per_prompt": config.generations_per_prompt,
                "epochs": config.epochs,
                "temperature": config.temperature,
                "learning_rate": config.learning_rate,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_targets": config.lora_targets,
                "max_prompt_tokens": config.max_prompt_tokens,
                "max_completion_tokens": config.max_completion_tokens,
                "thinking_budget": config.thinking_budget,
                "entropy_every_steps": config.entropy_every_steps,
                "clip_epsilon": config.clip_epsilon,
                "kl_beta": config.kl_beta,
                "batch_size": "adaptive",
                "batch_start_prompts": START_PROMPTS,
                "max_prompts": max_prompts,
                "gradient_accumulation": config.generations_per_prompt,
                "optimizer_updates_per_step": 1,
                "memory_floor_mib": config.memory_floor_mib,
                "test_workers": config.test_workers,
                "eval_batch_size": config.eval_batch_size,
                "prefetch_scoring": config.prefetch_scoring,
            },
            "checkpoint_dir": str(self.store.checkpoints_dir),
        }

    def _record_control(self, step: int, state: ControlState) -> None:
        """Write one event when the control file changes, never every step."""
        fingerprint = (frozenset(state.drop), tuple(sorted(state.tests.items())), self.control.error)
        if fingerprint == getattr(self, "_control_fingerprint", None):
            return
        self._control_fingerprint = fingerprint
        if self.control.error is not None:
            self.store.append_control_event(
                {"step": step, "action": "error", "error": self.control.error}
            )
        if state.drop or state.tests:
            self.store.append_control_event(
                {
                    "step": step,
                    "action": "apply",
                    "drop": sorted(state.drop),
                    "tests": sorted(state.tests),
                }
            )

    def _check_arm(self) -> None:
        """Reject an unknown arm, and an arm that has no reward yet.

        Called before `meta.json` is written, so a pending arm leaves no run
        folder and touches no backend.
        """
        check_arm_implemented(self.arm)

    def _pool_prompts(self) -> tuple[list[Prompt], list[Prompt]]:
        """The train and parked prompts the run holds.

        With a pool, only the `train` and `parked` ids are kept, in the order
        the caller gave them. Retired ids are never held. Without a pool,
        every prompt is train and nothing is parked.
        """
        if self.pools is None:
            return list(self.prompts), []
        train_ids = set(self.pools.train)
        parked_ids = set(self.pools.parked)
        train = [prompt for prompt in self.prompts if prompt.prompt_id in train_ids]
        parked = [prompt for prompt in self.prompts if prompt.prompt_id in parked_ids]
        return train, parked

    def _build_schedule(self) -> Schedule | None:
        """The configured schedule, or None for the placeholder path.

        A schedule needs a measured pool, because parked prompts come from the
        screen. The prompt count is the rollout budget divided by the group
        size. The spread parked block is half of that count. The sweep gap is
        the configured multiple of its own length.
        """
        policy = self.config.pool_policy
        if policy is None or self.pools is None:
            return None
        group = self.config.generations_per_prompt
        prompts_per_step = max(1, self.config.rollouts_per_step // group)
        if policy == "spread":
            return Spread(
                prompts_per_step=prompts_per_step,
                seed=self.config.seed,
                promotion_bar=DEFAULT_PROMOTION_BAR,
            )
        if policy == "sweep":
            return Sweep(
                prompts_per_step=prompts_per_step,
                gap_ratio=self.config.sweep_gap_ratio,
                seed=self.config.seed,
                promotion_bar=DEFAULT_PROMOTION_BAR,
            )
        # The config loader rejects an unknown policy, so this is unreachable
        # unless a caller builds a Config by hand.
        raise ValueError(f"unknown pool policy {policy!r}")

    def _pool_policy_name(self) -> str | None:
        """The policy name written to the step row and the run record."""
        if self._schedule is None:
            return None
        return getattr(self._schedule, "name", self.config.pool_policy)

    def _load_labels(self) -> None:
        """Replay pool changes and failure streaks from the run's label file.

        The file holds one row per change, one row per saved train streak and
        one row per step that saves both queues, in step order. The last row for
        a prompt wins. A row with an unknown label or streak is ignored, so a
        future row name cannot move a prompt by accident. A sweep position row
        is replayed last, so a resume mid sweep continues at the saved cycle
        index. The frozen sweep length is replayed too, so a resume keeps the
        number of sweep steps planned at the sweep start. A queues row is
        reconciled with the current pool before the schedule sees it.
        """
        reader = getattr(self.store, "read_labels", None)
        if reader is None:
            raise TypeError(
                "a scheduled run needs a store with read_labels, so a resume "
                "keeps the pool labels"
            )
        restore = getattr(self._schedule, "restore_streak", None)
        restore_position = getattr(self._schedule, "restore_position", None)
        restore_steps = getattr(self._schedule, "restore_sweep_steps", None)
        restore_order = getattr(self._schedule, "restore_order", None)
        restore_state = getattr(self._schedule, "restore_state", None)
        for row in reader():
            parsed = label_from_row(row)
            if parsed is not None:
                prompt_id, label = parsed
                self._move_prompt(prompt_id, to_train=label == "train")
                # A pool move starts the prompt with a clean streak. An older
                # file has no explicit zero row, so clear it here.
                if restore is not None:
                    restore(prompt_id, 0)
                self._saved_streaks[prompt_id] = 0
                continue
            saved_streak = streak_from_row(row)
            if saved_streak is not None and restore is not None:
                prompt_id, streak = saved_streak
                restore(prompt_id, streak)
                self._saved_streaks[prompt_id] = streak
                continue
            saved_steps = sweep_steps_from_row(row)
            if saved_steps is not None and restore_steps is not None:
                restore_steps(saved_steps)
            saved_position = sweep_position_from_row(row)
            if saved_position is not None and restore_position is not None:
                restore_position(saved_position)
            saved_order = sweep_order_from_row(row)
            if saved_order is not None and restore_order is not None:
                restore_order(saved_order)
            saved_queues = queues_from_row(row)
            if saved_queues is not None and restore_state is not None:
                parked_ids, train_ids = saved_queues
                restore_state(
                    restore_queue(
                        parked_ids,
                        (prompt.prompt_id for prompt in self._parked_prompts),
                    ),
                    restore_queue(
                        train_ids,
                        (prompt.prompt_id for prompt in self.prompts),
                    ),
                )

    def _observe_step(self, step: int, result: StepResult) -> None:
        """Ask the schedule for label changes, then apply and record them.

        Every train group's zero-pass count is saved after the label changes,
        so a resume continues counting instead of starting again at zero.
        """
        if self._schedule is None or not result.outcomes:
            return
        updates = self._schedule.observe(step, result.outcomes)
        self._apply_label_updates(updates)
        self._record_streaks(step, result.outcomes)

    def _record_schedule_state(self, step: int) -> None:
        """Save a schedule's state beside the label state.

        The row is the schedule's own shape. A schedule with a moving cycle
        saves its position. A schedule with queues saves both queues as a
        `queues` row. The row goes to the label file after the step, so a
        resume at this step's checkpoint replays the state that the next step
        should use.
        """
        if self._schedule is None:
            return
        row_for = getattr(self._schedule, "state_row", None)
        if row_for is None:
            return
        writer = getattr(self.store, "append_label", None)
        if writer is None:
            raise TypeError(
                "a scheduled run needs a store with append_label, so a resume "
                "keeps the schedule position"
            )
        writer(row_for(step))

    def _record_streaks(
        self, step: int, outcomes: Sequence[GroupOutcome]
    ) -> None:
        """Append a train count row when the count changes.

        A passing train group leaves the count alone, so it writes no row. A
        zero-pass group adds one and a demotion resets the count, so each writes
        one row. An unchanged count is already saved, so a resume restores it.
        """
        streak_for = getattr(self._schedule, "failure_streak", None)
        if streak_for is None:
            return
        writer = getattr(self.store, "append_label", None)
        if writer is None:
            raise TypeError(
                "a scheduled run needs a store with append_label, so a resume "
                "keeps the pool labels"
            )
        for outcome in outcomes:
            if outcome.parked:
                continue
            streak = streak_for(outcome.prompt_id)
            if self._saved_streaks.get(outcome.prompt_id, 0) == streak:
                continue
            writer(
                {
                    "step": step,
                    "prompt_id": outcome.prompt_id,
                    "streak": streak,
                }
            )
            self._saved_streaks[outcome.prompt_id] = streak

    def _apply_label_updates(self, updates: Sequence[LabelUpdate]) -> None:
        """Write each label change, then move the prompt to its new pool."""
        writer = getattr(self.store, "append_label", None)
        if updates and writer is None:
            raise TypeError(
                "a scheduled run needs a store with append_label, so a resume "
                "keeps the pool labels"
            )
        tracks_streak = getattr(self._schedule, "failure_streak", None) is not None
        for update in updates:
            writer(update.to_dict())
            self._move_prompt(update.prompt_id, to_train=update.label == "train")
            # A pool move starts the failure streak again at zero. Write the
            # zero so a resume does not restore an older streak for this prompt.
            if tracks_streak:
                writer(
                    {
                        "step": update.step,
                        "prompt_id": update.prompt_id,
                        "streak": 0,
                    }
                )
                self._saved_streaks[update.prompt_id] = 0

    def _move_prompt(self, prompt_id: str, *, to_train: bool) -> None:
        """Move one prompt between the train list and the parked list."""
        source = self._parked_prompts if to_train else self.prompts
        target = self.prompts if to_train else self._parked_prompts
        for index, prompt in enumerate(source):
            if prompt.prompt_id != prompt_id:
                continue
            del source[index]
            if all(existing.prompt_id != prompt_id for existing in target):
                target.append(prompt)
            return

    def _pools_meta(self) -> dict[str, Any]:
        """The pool counts for the run record.

        A run without a measured pool trains every provided prompt. Its
        `samples` is null and `measured` is false, because no screen count backs
        the membership. A real pool records its measured sample count and
        `measured` true.
        """
        pools = self.pools
        if pools is None:
            return {
                "train": len(self.prompts),
                "retired": 0,
                "parked": 0,
                "samples": None,
                "measured": False,
            }
        return {
            "train": len(pools.train),
            "retired": len(pools.retired),
            "parked": len(pools.parked),
            "samples": pools.samples,
            "measured": True,
        }

    def _schedule_meta(self) -> dict[str, Any] | None:
        """The scheduled step size, parked count and planned parked share.

        Both policies record the same fields. The share is the planned value at
        the start of the run. It can move when prompts promote or demote, so it
        is a plan, not a measurement.
        """
        schedule = self._schedule
        if schedule is None:
            return None
        parked = len(self._parked_prompts)
        share_for = getattr(schedule, "planned_parked_share", None)
        if not callable(share_for):
            share = None
        elif isinstance(schedule, Sweep):
            share = share_for(parked)
        else:
            share = share_for()
        return {
            "policy": self._pool_policy_name(),
            "step_prompts": schedule.prompts_per_step,
            "parked_prompts": parked,
            "planned_parked_share": share,
        }

    def _prompts_for_step(self, count: int) -> list[Prompt]:
        """The next `count` training prompts, cycling through the list.

        The cursor moves by one batch, not by one prompt, and a checkpoint
        records it. That is what lets a resume start on a batch boundary.
        """
        size = len(self.prompts)
        start = self._prompt_cursor % size
        block = [self.prompts[(start + index) % size] for index in range(count)]
        self._prompt_cursor = start + count
        return block

    def _plan_step(self, step: int, policy: BatchSizePolicy) -> StepPlan:
        """The prompts for one step, from the schedule or the batch policy.

        A configured schedule owns the prompt count, because its block formula
        depends on the rollout budget. Without one, the batch policy keeps the
        old adaptive count and only train prompts are drawn.
        """
        if self._schedule is not None:
            return self._schedule.plan(
                step, tuple(self.prompts), tuple(self._parked_prompts)
            )
        count = policy.count(len(self.prompts))
        return StepPlan(train=tuple(self._prompts_for_step(count)))

    def _train_with_memory_guard(
        self,
        step: int,
        policy: BatchSizePolicy,
        *,
        thinking_budget: int | None = None,
    ) -> StepResult | None:
        """Retry CUDA allocation failures at a strictly smaller prompt count.

        Failed attempts restore the prompt cursor. A previously successful size
        may fail on longer completions, so the retry must also shrink when the
        policy has already settled.
        """
        cursor = self._prompt_cursor
        plan: StepPlan | None = None
        while True:
            self._prompt_cursor = cursor
            if plan is None:
                plan = self._plan_step(step, policy)
            try:
                return self._training_step(
                    step, plan, thinking_budget=thinking_budget
                )
            except Exception as error:
                if not is_cuda_out_of_memory(error):
                    raise
                self._prompt_cursor = cursor
                policy.settle(failed_prompts=len(plan.prompts))
                _empty_cuda_cache()
                if len(plan.prompts) == 1:
                    return None
                if self._schedule is not None:
                    # A schedule owns the step size, so the retry drops one
                    # prompt from the end instead of asking it again unchanged.
                    plan = self._shrink_plan(plan)
                else:
                    # The batch policy now hands out a smaller count.
                    plan = None

    @staticmethod
    def _shrink_plan(plan: StepPlan) -> StepPlan:
        """One prompt fewer, parked first, so an out-of-memory retry can fit."""
        if plan.parked:
            return StepPlan(train=plan.train, parked=plan.parked[:-1])
        return StepPlan(train=plan.train[:-1], parked=())

    def _training_step(
        self,
        step: int,
        plan: StepPlan,
        thinking_budget: int | None = None,
    ) -> StepResult:
        """Sample every prompt group in one call, then apply one update.

        The generator returns one group per prompt, in prompt order. Each group
        is scored and parsed on its own, and its rewards are accumulated on
        their own, so advantages stay inside a prompt group and never cross the
        step. Train and parked groups share the call and the optimizer step, so
        parked rollouts produce gradients too.

        The entropy flag is the same for every group in a step. The step value
        is the mean over the groups that returned one. The advantage share
        counts groups whose rewards hold more than one distinct value. The
        group spread and group SD average the per-group values, and a group
        with a missing reward is skipped.
        """
        prompts = list(plan.prompts)
        parked_ids = {prompt.prompt_id for prompt in plan.parked}
        started = perf_counter()
        measure_entropy = self._measures_entropy(step)
        batches = self.generator.generate_batch(
            list(prompts),
            n=self.config.generations_per_prompt,
            temperature=self.config.temperature,
            thinking_budget=thinking_budget,
        )
        generation_seconds = perf_counter() - started
        self._check_batch_shape(prompts, batches)

        records: list[RolloutRecord] = []
        group_rewards: list[tuple[float | None, ...]] = []
        entropies: list[float] = []
        mastered: list[Prompt] = []
        outcomes: list[GroupOutcome] = []
        advantage_groups = 0
        truncated = 0
        prompt_truncated = 0
        scoring_seconds = 0.0
        scoring_wait_seconds = 0.0
        update_seconds = 0.0

        # Only scoring runs on the worker. Model calls, gradient accumulation,
        # record writes and optimizer updates remain on the training thread.
        # One group ahead bounds discarded judge work if accumulation fails.
        pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlvr-scoring")
            if self.config.prefetch_scoring else None
        )
        future = None
        try:
            if pool is not None:
                future = pool.submit(
                    self._score_prompt_group, step, prompts[0], batches[0]
                )
            for index, (prompt, generations) in enumerate(zip(prompts, batches)):
                wait_started = perf_counter()
                scored = (
                    future.result() if future is not None else
                    self._score_prompt_group(step, prompt, generations)
                )
                scoring_wait_seconds += perf_counter() - wait_started
                scoring_seconds += scored.seconds
                if pool is not None and index + 1 < len(prompts):
                    future = pool.submit(
                        self._score_prompt_group,
                        step,
                        prompts[index + 1],
                        batches[index + 1],
                    )
                update_started = perf_counter()
                entropy = self.generator.accumulate(
                    prompt,
                    generations,
                    [record.reward for record in scored.records],
                    measure_entropy=measure_entropy,
                )
                update_seconds += perf_counter() - update_started
                records.extend(scored.records)
                passed = sum(
                    1
                    for record in scored.records
                    if all_tests_passed(record.tests_passed, record.tests_total)
                )
                outcomes.append(
                    GroupOutcome(
                        prompt_id=prompt.prompt_id,
                        passed=passed,
                        samples=len(scored.records),
                        parked=prompt.prompt_id in parked_ids,
                    )
                )
                if len({record.reward for record in scored.records}) > 1:
                    advantage_groups += 1
                group_rewards.append(
                    tuple(record.reward for record in scored.records)
                )
                truncated += sum(generation.truncated for generation in generations)
                prompt_truncated += sum(
                    generation.prompt_truncated for generation in generations
                )
                if scored.solved:
                    mastered.append(prompt)
                if entropy is not None:
                    entropies.append(entropy)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

        update_started = perf_counter()
        # Optimizer allocation failures can leave some parameters updated. They
        # cannot safely be retried as another sample from the previous policy.
        try:
            grad_norm = self.generator.apply_update()
            update_norm = getattr(self.generator, "last_update_norm", None)
        except Exception as error:
            if is_cuda_out_of_memory(error):
                raise RuntimeError(
                    "optimizer allocation failed; resume from the last checkpoint "
                    "because the update may be partial"
                ) from error
            raise
        update_seconds += perf_counter() - update_started
        timings = {
            "generation_seconds": generation_seconds,
            "scoring_seconds": scoring_seconds,
            "scoring_wait_seconds": scoring_wait_seconds,
            "update_seconds": update_seconds,
            "step_seconds": perf_counter() - started,
        }
        if self.fake:
            timings = {name: 0.0 for name in timings}
        group_spread, group_sd = _mean_group_signal(group_rewards)
        return StepResult(
            records=tuple(records),
            prompts=len(prompts),
            truncated=truncated,
            prompt_truncated=prompt_truncated,
            mean_entropy=sum(entropies) / len(entropies) if entropies else None,
            grad_norm=grad_norm,
            update_norm=update_norm,
            mastered=tuple(mastered),
            timings=timings,
            train_prompts=len(plan.train),
            parked_prompts=len(plan.parked),
            advantage_share=advantage_groups / len(prompts) if prompts else 0.0,
            group_spread=group_spread,
            group_sd=group_sd,
            outcomes=tuple(outcomes),
        )

    def _check_batch_shape(
        self,
        prompts: Sequence[Prompt],
        batches: Sequence[Sequence[Generation]],
    ) -> None:
        """Reject a generator that does not return one group per prompt.

        The scoring path pairs groups with prompts by position, so a missing,
        extra or short group would score one prompt against another's output.
        The check runs before any test or judge call.
        """
        if len(batches) != len(prompts):
            raise ValueError(
                f"generate_batch returned {len(batches)} groups for "
                f"{len(prompts)} prompts"
            )
        expected = self.config.generations_per_prompt
        for prompt, group in zip(prompts, batches):
            if len(group) != expected:
                raise ValueError(
                    f"generate_batch returned {len(group)} generations for "
                    f"prompt {prompt.prompt_id!r}, expected {expected}"
                )

    def _prompt_group(
        self,
        step: int,
        prompt: Prompt,
        generations: Sequence[Generation],
        measure_entropy: bool,
    ) -> tuple[list[RolloutRecord], int, int, float | None, bool]:
        """Score and accumulate one group through the synchronous seam."""
        scored = self._score_prompt_group(step, prompt, generations)
        entropy = self.generator.accumulate(
            prompt,
            generations,
            [record.reward for record in scored.records],
            measure_entropy=measure_entropy,
        )
        return (
            list(scored.records),
            sum(generation.truncated for generation in generations),
            sum(generation.prompt_truncated for generation in generations),
            entropy,
            scored.solved,
        )

    def _run_tests(self, items: Sequence[tuple[Prompt, str]]) -> list[Outcome]:
        """Use a runner's bounded batch seam, with legacy runner support."""
        run_batch = getattr(self.test_runner, "run_batch", None)
        outcomes = (
            list(run_batch(items)) if run_batch is not None else
            [self.test_runner.run(prompt, code) for prompt, code in items]
        )
        if len(outcomes) != len(items):
            raise ValueError("test runner returned a different number of outcomes")
        return outcomes

    def _score_prompt_group(
        self,
        step: int,
        prompt: Prompt,
        generations: Sequence[Generation],
    ) -> _ScoredGroup:
        """Run tests and judges without touching the generator or store."""
        started = perf_counter()
        outcomes = self._run_tests(
            [(prompt, generation.code) for generation in generations]
        )
        passing = [
            index
            for index, outcome in enumerate(outcomes)
            if all_tests_passed(outcome.tests_passed, outcome.tests_total)
        ]

        # A prompt counts as solved only when every generation passed. One pass
        # in eight says the prompt is winnable by luck, not that the policy
        # holds the skill.
        solved = bool(outcomes) and len(passing) == len(outcomes)

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

        return _ScoredGroup(
            records=tuple(records), solved=solved, seconds=perf_counter() - started
        )

    def _measures_entropy(self, step: int) -> bool:
        """Whether this step is an entropy measurement step.

        `entropy_every_steps` counts training steps, so step N is measured
        when N is a multiple of the setting. Every prompt group accumulated in
        that step is measured. Zero disables the measurement.
        """
        every = self.config.entropy_every_steps
        return every > 0 and step % every == 0

    def _memory_report(self) -> MemoryReport:
        """The allocator figures for the step that just finished.

        Read once after the step's update, so the number covers the whole batch.
        Zero when no reader was injected, which is the fake path and every unit
        test that does not care about memory.
        """
        reader = self.memory
        if reader is None:
            return MemoryReport.zeros()
        return reader()

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
        # An empty or whitespace-only code field is the no-code case. The Docker
        # runner reports it as a runtime error, but the reward treats it like a
        # syntax error because no code was produced.
        code_error_class = (
            SYNTAX_ERROR_CLASS if not generation.code.strip() else outcome.error_class
        )

        reward = compose_reward(
            self.arm,
            tests_passed=outcome.tests_passed,
            tests_total=outcome.tests_total,
            judge_score=judge_score,
            process_score=process_score,
            truncated=generation.truncated,
            closed=generation.closed,
            error_class=code_error_class,
            completion_tokens=generation.completion_tokens,
            length_soft_limit=self.config.length_soft_limit,
            length_penalty_per_token=self.config.length_penalty_per_token,
            length_penalty_floor=self.config.length_penalty_floor,
        )

        # A question with no tests cannot open the gate. The gated arms record a
        # code score of 0.0 and reward 0.0, with no judge call. The code-score
        # arm still fails loudly: there, zero tests is a broken test runner.
        # The exception is a syntax error: the code never ran, so there are no
        # counts to divide, and the code arm records 0.0 alongside its -0.5.
        if outcome.tests_total == 0 and gates_on_tests(self.arm):
            record_code_score = 0.0
        elif outcome.tests_total == 0 and code_error_class == SYNTAX_ERROR_CLASS:
            record_code_score = 0.0
        else:
            record_code_score = code_score(outcome.tests_passed, outcome.tests_total)

        return RolloutRecord(
            arm=self.arm,
            step=step,
            stage=UNSTAGED_STAGE,
            prompt_id=prompt.prompt_id,
            reasoning=generation.reasoning,
            code=generation.code,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
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
        """Ask the generator for its state, then write the checkpoint manifest.

        The manifest carries the prompt cursor and the batch policy state. A
        resume therefore continues on the next batch boundary with the same
        size, instead of rewinding the prompt list. It does not carry a stage
        position, because the run has one pool.

        After the manifest is written, older checkpoints beyond the retained
        set are deleted. This only touches this run folder, never persistent
        storage.
        """
        directory = self.store.checkpoints_dir / checkpoint_dir_name(step)
        self.generator.save_checkpoint(directory)
        batch = None
        if self._policy is not None:
            batch = {
                "current": self._policy.current,
                "largest_fit": self._policy.largest_fit,
                "settled": self._policy.settled,
            }
        write_checkpoint_manifest(
            directory,
            step=step,
            arm=self.arm,
            timestamp=self.store.timestamp,
            prompt_cursor=self._prompt_cursor,
            batch=batch,
        )
        existing: list[int] = []
        for child in self.store.checkpoints_dir.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            parsed = parse_checkpoint_step(child.name)
            if parsed is not None:
                existing.append(parsed)
        # Keep the checkpoint just written even if a reused folder already
        # holds higher step numbers.
        keep = set(select_checkpoints_to_keep([*existing, step])) | {step}
        prune_checkpoints(self.store.checkpoints_dir, keep)

    @staticmethod
    def _metric_row(
        step: int,
        records: Sequence[RolloutRecord],
        *,
        prompts: int,
        truncated: int = 0,
        prompt_truncated: int = 0,
        mean_entropy: float | None = None,
        grad_norm: float | None = None,
        update_norm: float | None = None,
        memory: MemoryReport | None = None,
        timings: Mapping[str, float] | None = None,
        max_prompts: int | None = None,
        pool_policy: str | None = None,
        train_prompts: int = 0,
        parked_prompts: int = 0,
        advantage_share: float = 0.0,
        group_spread: float = 0.0,
        group_sd: float = 0.0,
    ) -> dict[str, Any]:
        """One metrics row. Entropy is present only on a measurement step.

        `prompts` is the prompt groups in the batch and `sequences` is the
        sampled completions across them. `rollouts` keeps the old name for the
        sequence count, so an older reader keeps working. `train_prompts` and
        `parked_prompts` split the group count by the plan. `advantage_share`
        is the share of groups with more than one distinct reward, so a step
        that teaches nothing is visible in the row. `group_spread` and
        `group_sd` average the per-group reward spread and population SD, so
        the row shows how much there is to learn from. `grad_norm` and
        `update_norm` are null when the generator has no gradients, as the fake
        does, or when a diverged step produced a non-finite value.
        """
        count = len(records)
        passed = sum(
            1
            for record in records
            if all_tests_passed(record.tests_passed, record.tests_total)
        )
        row: dict[str, Any] = {
            "step": step,
            "arm": records[0].arm if records else None,
            "stage": UNSTAGED_STAGE,
            "pool_policy": pool_policy,
            "prompts": prompts,
            "train_prompts": train_prompts,
            "parked_prompts": parked_prompts,
            "sequences": count,
            "rollouts": count,
            "max_prompts": max_prompts,
            "mean_reward": sum(record.reward for record in records) / count,
            "pass_rate": passed / count,
            "advantage_share": advantage_share,
            "group_spread": group_spread,
            "group_sd": group_sd,
            "mean_completion_tokens": sum(record.completion_tokens for record in records) / count,
            "cache_hits": sum(1 for record in records if record.cache_hit),
            "truncation_rate": truncated / count,
            "prompt_truncation_rate": prompt_truncated / count,
            "grad_norm": _finite_or_none(grad_norm),
            "update_norm": _finite_or_none(update_norm),
        }
        row.update(timings or {})
        seconds = row.get("step_seconds", 0.0)
        if seconds > 0.0:
            row["sequences_per_second"] = count / seconds
            row["completion_tokens_per_second"] = (
                sum(record.completion_tokens for record in records) / seconds
            )
        row.update((memory or MemoryReport.zeros()).as_metrics())
        if mean_entropy is not None:
            row["mean_entropy"] = mean_entropy
        return row

    def _evaluate(
        self,
        step: int,
        prompts: Sequence[Prompt],
        *,
        kind: str,
        subset: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Greedy pass@1 in bounded batches, shrinking on CUDA OOM."""
        started = perf_counter()
        prompt_ids: list[str] = []
        passed: list[bool] = []
        batch_size = self.config.eval_batch_size
        offset = 0
        while offset < len(prompts):
            chunk = list(prompts[offset : offset + batch_size])
            try:
                groups = self.generator.generate_batch(
                    chunk, n=1, temperature=EVAL_TEMPERATURE
                )
            except Exception as error:
                if not is_cuda_out_of_memory(error) or len(chunk) == 1:
                    raise
                _empty_cuda_cache()
                batch_size = max(1, len(chunk) // 2)
                continue
            if len(groups) != len(chunk) or any(len(group) != 1 for group in groups):
                raise ValueError("evaluation needs exactly one generation per prompt")
            outcomes = self._run_tests(
                [(prompt, group[0].code) for prompt, group in zip(chunk, groups)]
            )
            prompt_ids.extend(prompt.prompt_id for prompt in chunk)
            passed.extend(
                all_tests_passed(result.tests_passed, result.tests_total)
                for result in outcomes
            )
            offset += len(chunk)
        point: dict[str, Any] = {
            "step": step,
            "kind": kind,
            "stage": UNSTAGED_STAGE,
            "evaluation_seconds": 0.0 if self.fake else perf_counter() - started,
            "pass_at_1": pass_at_1(passed),
            "count": len(prompt_ids),
            "prompt_ids": prompt_ids,
            "subset": dict(subset) if subset is not None else None,
        }
        return point

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
