"""Command line interface.

    rlvr run --arm <code-score|hybrid|gated-judge|gated-process> --fake
    rlvr summary <run-dir|runs-root> [...] [--html report.html] [--no-html]
    rlvr dashboard [<run-dir> ...] [--runs-root <runs-root>] [--port 8081]
    rlvr publish --report report.html --remote <url-or-path> [--branch gh-pages]

This build runs both the fake path and, on a GPU host with the `train` extra
installed, the real Qwen generator, Docker runner, and GRPO trainer. The judge
backend is chosen before the run folder is made, so a judge arm with no key
fails at startup. `--resume` continues the latest run folder from its last
checkpoint.

`summary` reads run folders that already exist. It prints one table per metric
with arms side by side and writes a static, self-contained HTML report.
`publish` force-pushes that report to a Pages branch. It needs a remote the
caller's git can already write to.

Exit codes: 0 success, 2 for a bad command line, unknown or unimplemented arm,
bad config, a missing judge key, a bad run folder, or a failed publish. Messages
for those cases go to stderr without a traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import Config, ConfigError, default_config_path, load_config
from .checkpoint import find_latest_checkpoint
from .data import (
    TEST_SIZE,
    TRAIN_SIZE,
    DatasetError,
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
from .stages import (
    BANDS,
    LabelledRow,
    Stage,
    StageError,
    build_stages as build_ordered_stages,
    load_band_labels,
)
from .summary import SummaryError, format_summary, load_reports, write_report

__all__ = [
    "build_curriculum",
    "build_judge",
    "build_parser",
    "build_prompts",
    "build_test_runner",
    "main",
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


def build_prompts(config: Config) -> tuple[list[Prompt], list[Prompt]]:
    """Load the fixed split and return the train and test prompts.

    Training uses the 714 train questions and evaluation uses the 250 test
    questions. The split and the prompt format come from `data.py`, so every
    arm and the fake path see the same questions. The two lists never share a
    question. A wrong or empty split stops the run before a folder is made.
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

    train_prompts = prompts_for_questions(split.train)
    eval_prompts = prompts_for_questions(split.test)
    if not train_prompts or not eval_prompts:
        raise UsageError("the split produced an empty train or test prompt list")
    return train_prompts, eval_prompts


def build_curriculum(
    config: Config, prompts: list[Prompt], *, fake: bool
) -> tuple[Stage[Prompt], ...]:
    """Build the ordered stages for the training prompts.

    A real run reads the difficulty label file. The loader returns an empty
    mapping when the file is absent, and unlabelled prompts fall back to the
    configured start band, so this task does not block on the label data.

    The fake path supplies its own deterministic labels. The first fake prompt
    that fully passes is put alone in the first band, so the fake run shows a
    stage one exhaustion and a stage two start at a fixed step.
    """
    if fake:
        labels = fake_band_labels(prompts)
    else:
        try:
            labels = load_band_labels()
        except StageError as error:
            raise UsageError(f"cannot load difficulty labels: {error}") from error

    rows = [
        LabelledRow(
            prompt_id=prompt.prompt_id,
            band=labels.get(prompt.prompt_id, config.stage),
            item=prompt,
        )
        for prompt in prompts
    ]
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
    """The Docker runner over the full split, or the fake runner.

    The real runner holds every question because training rollouts run the
    train questions and evaluation runs the held-out test questions. The fake
    runner ignores questions entirely.
    """
    if fake:
        return FakeTestRunner()
    try:
        split = load_split(seed=config.split_seed)
    except DatasetError as error:
        raise UsageError(f"cannot load the MBPP split: {error}") from error
    questions = list(split.train) + list(split.test)
    from .docker_runner import DockerTestRunner

    return DockerTestRunner(questions)


def build_generator(config: Config, *, fake: bool):
    """The real Qwen trainer, or the fake generator."""
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
        )
    except ImportError as error:
        raise UsageError(
            "the real generator needs torch, transformers, and peft. "
            f"install the train extra or pass --fake. ({error})"
        ) from error


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

    # Real runs sweep the train questions once per epoch, one prompt group per
    # step, so the step budget is epochs times the train size. The fake path
    # keeps the small `[fake] steps` count.
    if not args.fake:
        from dataclasses import replace

        config = replace(config, steps=config.epochs * len(train_prompts))

    resume_step = 0
    prior_evaluations: list[dict] = []
    checkpoint = None
    if args.resume:
        store = RunStore.resume(args.runs_root, args.arm)
        checkpoint = find_latest_checkpoint(store.checkpoints_dir)
        if checkpoint is None:
            raise UsageError(
                f"no checkpoint found for arm {args.arm!r} under {store.run_dir}"
            )
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
