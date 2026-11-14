"""Command line interface.

    rlvr run --arm <code-score|hybrid|gated-judge|gated-process> --fake
    rlvr screen --questions <path> --samples <n> --output <dir> [--fake]
    rlvr summary <run-dir|runs-root> [...] [--html report.html] [--no-html]
    rlvr dashboard [<run-dir> ...] [--runs-root <runs-root>] [--port 8081]
    rlvr publish --report report.html --remote <url-or-path> [--branch gh-pages]

This build runs both the fake path and, on a GPU host with the `train` extra
installed, the real Qwen generator, Docker runner, and GRPO trainer. The judge
backend is chosen before the run folder is made, so a judge arm with no key
fails at startup. `--resume` continues the latest run folder from its last
checkpoint.

`screen` samples the base model with thinking off and writes a difficulty table.
It never trains, builds no optimizer, and writes into its own output folder, not
a run folder. A restart skips the question ids already in the table.

`summary` reads run folders that already exist. It prints one table per metric
with arms side by side and writes a static, self-contained HTML report.
`publish` force-pushes that report to a Pages branch. It needs a remote the
caller's git can already write to.

Exit codes: 0 success, 2 for a bad command line, unknown or unimplemented arm,
bad config, a missing judge key, a bad run folder, a bad question set, or a
failed publish. Messages for those cases go to stderr without a traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .batching import START_PROMPTS, BatchSizePolicy
from .config import Config, ConfigError, default_config_path, load_config
from .checkpoint import find_latest_checkpoint, read_checkpoint_manifest
from .data import (
    TEST_SIZE,
    TRAIN_SIZE,
    DatasetError,
    MBPPQuestion,
    load_dataset,
    load_difficulty_labels,
    load_split,
    prompts_for_questions,
)
from .experiment import Experiment, Judge, JudgeResult, Prompt
from .fake import (
    FakeGenerator,
    FakeJudge,
    FakeTestRunner,
    fake_band_labels,
    fake_memory,
)
from .judge import JudgeClient, MissingAPIKeyError
from .memory import configure_expandable_segments, read_torch_memory
from .publish import PublishError, publish_report
from .reward import (
    ARMS,
    ArmNotImplemented,
    check_arm_implemented,
    uses_output_judge,
    uses_process_judge,
)
from .runs import RunStore
from .screening import ScreenError, screen as run_screen
from .stages import (
    BANDS,
    LabelledRow,
    Stage,
    StageError,
    build_stages as build_ordered_stages,
)
from .summary import SummaryError, format_summary, load_reports, write_report
from .synthetic import SyntheticError, load_synthetic_questions

__all__ = [
    "build_curriculum",
    "build_generator",
    "build_judge",
    "build_parser",
    "build_prompts",
    "build_screen_test_runner",
    "build_test_runner",
    "load_synthetic_rows",
    "main",
    "screen_command",
]

USAGE_ERROR = 2


class UsageError(ValueError):
    """A command line problem the user can fix."""


class UnusedJudge:
    """A judge for arms that never score with one. Calling it is a bug."""

    def score_output(self, prompt: Prompt, code: str) -> JudgeResult:
        raise AssertionError("this arm must never call the output judge")

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult:
        raise AssertionError("this arm must never call the process judge")

    def score_outputs(self, items) -> list[JudgeResult]:
        raise AssertionError("this arm must never call the output judge")

    def score_processes(self, items) -> list[JudgeResult]:
        raise AssertionError("this arm must never call the process judge")


def _resume_batch_policy(config: Config, manifest: dict) -> BatchSizePolicy | None:
    """The saved batch policy, or None for an older manifest.

    A manifest written before batching has no `batch` table. A corrupt table is
    ignored so the run still resumes from the step, which is the part that
    matters most.
    """
    batch = manifest.get("batch")
    if not isinstance(batch, dict):
        return None
    current = batch.get("current")
    largest_fit = batch.get("largest_fit", 0)
    settled = batch.get("settled", False)
    if (
        isinstance(current, bool)
        or not isinstance(current, int)
        or current < 1
        or isinstance(largest_fit, bool)
        or not isinstance(largest_fit, int)
        or largest_fit < 0
        or not isinstance(settled, bool)
    ):
        return None
    return BatchSizePolicy(
        floor_mib=config.memory_floor_mib,
        current=current,
        largest_fit=largest_fit,
        settled=settled,
    )


def _resume_stage(manifest: dict) -> tuple[int | None, int, list[str] | None]:
    """The saved curriculum position, or defaults for an older manifest.

    `None` for the index and the pool means "start the curriculum from the
    beginning", which is what a pre-batching manifest gets.
    """
    stage = manifest.get("stage")
    if not isinstance(stage, dict):
        return None, 0, None
    index = stage.get("index")
    step = stage.get("step", 0)
    pool = stage.get("pool")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return None, 0, None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None, 0, None
    if not isinstance(pool, list) or not all(isinstance(item, str) for item in pool):
        return index, step, None
    return index, step, list(pool)


def _manifest_count(manifest: dict, name: str, default: int) -> int:
    """A non-negative checkpoint manifest field, or the default.

    A field absent from an older manifest falls back to the default. A corrupt
    value does the same rather than stopping the resume, because the step and
    arm are the only fields a checkpoint needs to be usable.
    """
    value = manifest.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def build_judge(arm: str, config: Config, *, fake: bool) -> Judge:
    """Pick the judge backend for an arm, before the run folder is created.

    The fake path stays offline and needs no key. A real run builds the judge
    client only for arms that score with it, so a missing key fails at startup
    instead of after the first rollout. The code-score arm gets a judge that
    raises if it is ever called.
    """
    if fake:
        return FakeJudge()
    if uses_output_judge(arm) or uses_process_judge(arm):
        return JudgeClient.from_config(config)
    return UnusedJudge()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlvr", description="Small-model code RLVR study")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one arm")
    run.add_argument("--arm", required=True, help=f"one of: {', '.join(ARMS)}")
    run.add_argument(
        "--fake",
        action="store_true",
        help="use the deterministic fake generator, judge, and test runner",
    )
    run.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config file to read (default: the packaged config)",
    )
    run.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="folder that holds run folders (default: ./runs)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue the latest run folder from its last checkpoint",
    )

    screen_parser = subparsers.add_parser(
        "screen",
        help="sample the base model and write a difficulty table",
    )
    screen_parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        metavar="PATH",
        help="question set in the MBPP JSONL row schema",
    )
    screen_parser.add_argument(
        "--samples",
        type=int,
        required=True,
        metavar="N",
        help="samples per question",
    )
    screen_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="folder for rows.jsonl and summary.json",
    )
    screen_parser.add_argument(
        "--fake",
        action="store_true",
        help="use the deterministic fake generator and test runner",
    )
    screen_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config file to read (default: the packaged config)",
    )

    summary = subparsers.add_parser(
        "summary", help="print run metrics and write the static HTML report"
    )
    summary.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="RUN",
        help="run folders, arm folders, or a runs root",
    )
    summary.add_argument(
        "--html",
        type=Path,
        default=Path("report.html"),
        help="where to write the report (default: ./report.html)",
    )
    summary.add_argument(
        "--no-html",
        action="store_true",
        help="print the tables and skip the HTML report",
    )

    dashboard = subparsers.add_parser(
        "dashboard", help="serve a live progress dashboard in the browser"
    )
    dashboard.add_argument(
        "paths",
        nargs="*",
        type=Path,
        metavar="RUN",
        help="run folders or arm folders to watch",
    )
    dashboard.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="runs root that expands to each arm's latest run",
    )
    dashboard.add_argument(
        "--port",
        type=int,
        default=8081,
        help="port to serve on (default: 8081)",
    )

    publish = subparsers.add_parser(
        "publish", help="force-push the HTML report to a GitHub Pages branch"
    )
    publish.add_argument(
        "--report",
        type=Path,
        default=Path("report.html"),
        help="report to publish (default: ./report.html)",
    )
    publish.add_argument(
        "--remote",
        required=True,
        help="git remote URL or path, for example git@github.com:<user>/<repo>.git",
    )
    publish.add_argument(
        "--branch",
        default="gh-pages",
        help="branch to write (default: gh-pages). It is replaced each time.",
    )
    publish.add_argument(
        "--message",
        default=None,
        help="commit message for the published report",
    )
    return parser


def load_synthetic_rows() -> tuple[MBPPQuestion, ...]:
    """The 50 synthetic very easy questions as runner rows.

    The synthetic file is committed beside the MBPP data. A missing or invalid
    file stops the run before a folder is made, because the very easy stage has
    no other material.
    """
    try:
        return tuple(
            question.as_question() for question in load_synthetic_questions()
        )
    except SyntheticError as error:
        raise UsageError(f"cannot load the synthetic questions: {error}") from error


def build_prompts(config: Config) -> tuple[list[Prompt], list[Prompt]]:
    """Load the fixed split and return the train and test prompts.

    Training uses the 714 MBPP train questions plus the 50 synthetic very easy
    questions. Evaluation uses the 250 test questions. The split and the prompt
    format come from `data.py`, so every arm and the fake path see the same
    questions. The two lists never share a question, and a synthetic id can
    never collide with an MBPP id. A wrong or empty split stops the run before
    a folder is made.
    """
    try:
        split = load_split(seed=config.split_seed)
    except DatasetError as error:
        raise UsageError(f"cannot load the MBPP split: {error}") from error

    if len(split.train) != TRAIN_SIZE or len(split.test) != TEST_SIZE:
        raise UsageError(
            f"the split must be {TRAIN_SIZE} train and {TEST_SIZE} test questions, "
            f"got {len(split.train)} and {len(split.test)}"
        )
    train_ids = {question.task_id for question in split.train}
    test_ids = {question.task_id for question in split.test}
    if train_ids & test_ids:
        raise UsageError(
            f"the train and test split overlap on {len(train_ids & test_ids)} task ids"
        )

    train_questions = list(split.train) + list(load_synthetic_rows())
    train_prompts = prompts_for_questions(train_questions)
    eval_prompts = prompts_for_questions(split.test)
    if not train_prompts or not eval_prompts:
        raise UsageError("the split produced an empty train or test prompt list")
    return train_prompts, eval_prompts


def build_curriculum(
    config: Config, prompts: list[Prompt], *, fake: bool
) -> tuple[Stage[Prompt], ...]:
    """Build the ordered stages for the training prompts.

    A real run validates the difficulty label file against the MBPP split for
    the configured seed. A missing file, an unknown or duplicated id, or labels
    written for a different split seed all stop the run with a clear message,
    so a wrong config cannot put every row in one band.

    The 50 synthetic rows are always placed in the very easy band. They carry
    their own tests and exist to make the first stage solvable.

    The fake path supplies its own deterministic labels. The first fake prompt
    that fully passes is put alone in the first band, so the fake run shows a
    stage one exhaustion and a stage two start at a fixed step.
    """
    if fake:
        labels: Mapping[str, str] = fake_band_labels(prompts)
        synthetic_ids: set[str] = set()
    else:
        try:
            split = load_split(seed=config.split_seed)
            labels = load_difficulty_labels(split)
        except DatasetError as error:
            raise UsageError(f"cannot load difficulty labels: {error}") from error
        synthetic_ids = {question.prompt_id for question in load_synthetic_rows()}

    rows = []
    for prompt in prompts:
        if prompt.prompt_id in synthetic_ids:
            band = BANDS[0]
        else:
            band = labels.get(prompt.prompt_id)
            if band is None:
                raise UsageError(
                    f"prompt {prompt.prompt_id!r} has no difficulty label. "
                    f"The label file must cover the training split for seed "
                    f"{config.split_seed}."
                )
        rows.append(
            LabelledRow(prompt_id=prompt.prompt_id, band=band, item=prompt)
        )
    try:
        return build_ordered_stages(
            rows,
            BANDS,
            config.carry_over_ratio,
            seed=config.split_seed,
            start_stage=config.stage,
        )
    except StageError as error:
        raise UsageError(f"cannot build curriculum stages: {error}") from error


def build_test_runner(config: Config, *, fake: bool):
    """The Docker runner over the full split and the synthetic rows, or the fake.

    The real runner holds every question because training rollouts run the
    train questions and the synthetic easy questions, and evaluation runs the
    held-out test questions and each stage's held-out slice. The fake runner
    ignores questions entirely.
    """
    if fake:
        return FakeTestRunner()
    try:
        split = load_split(seed=config.split_seed)
    except DatasetError as error:
        raise UsageError(f"cannot load the MBPP split: {error}") from error
    questions = list(split.train) + list(split.test) + list(load_synthetic_rows())
    from .docker_runner import DockerTestRunner

    return DockerTestRunner(questions)


def build_generator(config: Config, *, fake: bool, trainable: bool = True):
    """The real Qwen generator, or the fake generator.

    `trainable` false loads the base model with no adapter and no optimizer.
    The difficulty screen uses that. Training keeps the default.
    """
    if fake:
        return FakeGenerator()
    from .qwen import QwenGenerator

    try:
        return QwenGenerator(
            model_name=config.model_name,
            revision=config.model_revision,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            learning_rate=config.learning_rate,
            max_prompt_tokens=config.max_prompt_tokens,
            max_completion_tokens=config.max_completion_tokens,
            clip_epsilon=config.clip_epsilon,
            kl_beta=config.kl_beta,
            seed=config.seed,
            trainable=trainable,
        )
    except ImportError as error:
        raise UsageError(
            "the real generator needs torch, transformers, and peft. "
            f"install the train extra or pass --fake. ({error})"
        ) from error


def build_screen_test_runner(questions, *, fake: bool):
    """The Docker runner over this question set, or the fake runner.

    The screen builds its runner from the same rows it samples, not from the
    fixed split, so a synthetic question set works without touching the data
    split.
    """
    if fake:
        return FakeTestRunner()
    from .docker_runner import DockerTestRunner

    return DockerTestRunner(questions)


def run_command(args: argparse.Namespace) -> int:
    # PyTorch reads PYTORCH_CUDA_ALLOC_CONF once, at allocator startup, so set
    # it before anything can import torch. A value the user already set wins.
    configure_expandable_segments()

    if args.arm not in ARMS:
        raise UsageError(f"unknown arm {args.arm!r}. valid arms: {', '.join(ARMS)}")

    config = load_config(args.config if args.config is not None else default_config_path())
    try:
        check_arm_implemented(args.arm)
    except ArmNotImplemented as error:
        raise UsageError(str(error)) from error

    # Build the judge before the run folder, so a missing key for a judge arm
    # fails before any rollout and leaves no run folder behind.
    judge = build_judge(args.arm, config, fake=args.fake)

    train_prompts, eval_prompts = build_prompts(config)
    stages = build_curriculum(config, train_prompts, fake=args.fake)
    if config.eval_final_questions is not None:
        eval_prompts = eval_prompts[: config.eval_final_questions]

    # Real runs sweep the train questions once per epoch. A step takes a batch
    # of prompts, so the budget is epochs times the number of batches at the
    # starting size. The fake path keeps the small `[fake] steps` count.
    if not args.fake:
        from dataclasses import replace

        batches_per_epoch = -(-len(train_prompts) // START_PROMPTS)
        config = replace(config, steps=config.epochs * batches_per_epoch)

    resume_step = 0
    resume_prompt_cursor = 0
    resume_stage_index: int | None = None
    resume_stage_step = 0
    resume_stage_pool: list[str] | None = None
    batch_policy: BatchSizePolicy | None = None
    prior_evaluations: list[dict] = []
    checkpoint = None
    if args.resume:
        store = RunStore.resume(args.runs_root, args.arm)
        checkpoint = find_latest_checkpoint(store.checkpoints_dir)
        if checkpoint is None:
            raise UsageError(
                f"no checkpoint found for arm {args.arm!r} under {store.run_dir}"
            )
        manifest = read_checkpoint_manifest(checkpoint.directory)
        resume_prompt_cursor = _manifest_count(manifest, "prompt_cursor", 0)
        resume_stage_index, resume_stage_step, resume_stage_pool = _resume_stage(
            manifest
        )
        batch_policy = _resume_batch_policy(config, manifest)
        store.truncate_to_step(checkpoint.step)
        prior_evaluations = store.read_evaluations()
        resume_step = checkpoint.step

    # Build the generator before the run folder is created, so a failed model
    # load leaves no empty folder behind. Resume loads the checkpoint afterwards.
    generator = build_generator(config, fake=args.fake)
    if args.resume:
        generator.load_checkpoint(checkpoint.directory)

    if not args.resume:
        try:
            store = RunStore(args.runs_root, args.arm)
        except OSError as error:
            raise UsageError(f"cannot create the run folder: {error}") from error

    test_runner = build_test_runner(config, fake=args.fake)

    experiment = Experiment(
        arm=args.arm,
        config=config,
        generator=generator,
        judge=judge,
        test_runner=test_runner,
        store=store,
        prompts=train_prompts,
        stages=stages,
        eval_prompts=eval_prompts,
        resume_step=resume_step,
        prior_evaluations=prior_evaluations,
        fake=args.fake,
        memory=fake_memory if args.fake else read_torch_memory,
        resume_prompt_cursor=resume_prompt_cursor,
        resume_stage_index=resume_stage_index,
        resume_stage_step=resume_stage_step,
        resume_stage_pool=resume_stage_pool,
        batch_policy=batch_policy,
    )
    summary = experiment.run()

    print(f"arm={summary.arm} steps={summary.steps} rollouts={summary.rollouts}")
    print(
        f"mean_reward={summary.mean_reward:.4f} pass_at_1={summary.pass_at_1:.4f} "
        f"stop_reason={summary.stop_reason} stage={summary.stage or '-'} "
        f"run folder: {summary.run_dir}"
    )
    if summary.stop_reason == "curriculum_complete":
        print(
            "curriculum complete: every stage pool was exhausted "
            f"after {summary.steps} steps"
        )
    return 0


def screen_command(args: argparse.Namespace) -> int:
    """Sample the base model and write the difficulty table."""
    if args.samples < 1:
        raise UsageError(f"--samples must be at least 1, got {args.samples}")

    config = load_config(args.config if args.config is not None else default_config_path())
    try:
        questions = load_dataset(args.questions)
    except DatasetError as error:
        raise UsageError(f"cannot load the question set: {error}") from error
    if not questions:
        raise UsageError(f"the question set is empty: {args.questions}")

    prompts = prompts_for_questions(questions)
    generator = build_generator(config, fake=args.fake, trainable=False)
    test_runner = build_screen_test_runner(questions, fake=args.fake)
    try:
        result = run_screen(
            prompts,
            samples=args.samples,
            directory=args.output,
            generator=generator,
            test_runner=test_runner,
            temperature=config.temperature,
        )
    except ScreenError as error:
        raise UsageError(str(error)) from error
    except OSError as error:
        raise UsageError(f"cannot write the screen output: {error}") from error

    summary = result.summary
    print(
        f"screen: questions={summary['questions']} samples={args.samples} "
        f"resumed={result.resumed} zero_pass={summary['zero_pass']} "
        f"partial_pass={summary['partial_pass']} full_pass={summary['full_pass']}"
    )
    print(f"rows: {result.directory / 'rows.jsonl'}")
    print(f"summary: {result.directory / 'summary.json'}")
    return 0


def summary_command(args: argparse.Namespace) -> int:
    """Print the metric tables and write the report."""
    try:
        reports = load_reports(args.paths)
    except SummaryError as error:
        raise UsageError(str(error)) from error

    print(format_summary(reports), end="")

    if not args.no_html:
        try:
            written = write_report(args.html, reports)
        except OSError as error:
            raise UsageError(f"cannot write the report: {error}") from error
        print(f"report: {written}")
    return 0


def dashboard_command(args: argparse.Namespace) -> int:
    """Serve the live dashboard until Ctrl+C."""
    paths = [str(path) for path in args.paths]
    if args.runs_root is not None:
        paths.append(str(args.runs_root))
    if not paths:
        raise UsageError("dashboard needs a run folder, an arm folder, or --runs-root")
    if not 0 < args.port <= 65535:
        raise UsageError(f"invalid port {args.port}")

    from .dashboard import serve_dashboard

    try:
        serve_dashboard(paths, host="127.0.0.1", port=args.port)
    except OSError as error:
        raise UsageError(f"cannot start the dashboard server: {error}") from error
    return 0


def publish_command(args: argparse.Namespace) -> int:
    """Push an existing report to the Pages branch."""
    try:
        commit = publish_report(
            args.report,
            remote=args.remote,
            branch=args.branch,
            message=args.message,
        )
    except PublishError as error:
        raise UsageError(str(error)) from error
    print(
        f"published {args.report} to {args.branch} on {args.remote} "
        f"at {commit[:12]}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_command(args)
        if args.command == "screen":
            return screen_command(args)
        if args.command == "summary":
            return summary_command(args)
        if args.command == "dashboard":
            return dashboard_command(args)
        if args.command == "publish":
            return publish_command(args)
        raise UsageError(f"unknown command {args.command!r}")
    except (UsageError, ConfigError, MissingAPIKeyError) as error:
        print(f"rlvr: {error}", file=sys.stderr)
        return USAGE_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
