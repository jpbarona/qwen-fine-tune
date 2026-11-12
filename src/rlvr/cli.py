"""Command line interface.

    rlvr run --arm <code-score|hybrid|gated-judge|gated-process> --fake
    rlvr summary <run-dir|runs-root> [...] [--html report.html] [--no-html]
    rlvr publish --report report.html --remote <url-or-path> [--branch gh-pages]

This build implements the fake path only. Real backends arrive in later tasks
and plug into the same Experiment seam without changing this parser. The judge
backend is chosen before the run folder is made, so a judge arm with no key
fails at startup.

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
from .data import (
    TEST_SIZE,
    TRAIN_SIZE,
    DatasetError,
    load_split,
    prompts_for_questions,
)
from .experiment import Experiment, Judge, JudgeResult, Prompt
from .fake import FakeGenerator, FakeJudge, FakeTestRunner
from .judge import JudgeClient, MissingAPIKeyError
from .publish import PublishError, publish_report
from .reward import (
    ARMS,
    ArmNotImplemented,
    check_arm_implemented,
    uses_output_judge,
    uses_process_judge,
)
from .runs import RunStore
from .summary import SummaryError, format_summary, load_reports, write_report

__all__ = ["build_judge", "build_parser", "build_prompts", "main"]

USAGE_ERROR = 2


class UsageError(ValueError):
    """A command line problem the user can fix."""


class UnusedJudge:
    """A judge for arms that never score with one. Calling it is a bug."""

    def score_output(self, prompt: Prompt, code: str) -> JudgeResult:
        raise AssertionError("this arm must never call the output judge")

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult:
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


def run_command(args: argparse.Namespace) -> int:
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

    if not args.fake:
        raise UsageError(
            f"real backends for arm {args.arm!r} are not implemented yet, add --fake"
        )

    train_prompts, eval_prompts = build_prompts(config)

    try:
        store = RunStore(args.runs_root, args.arm)
    except OSError as error:
        raise UsageError(f"cannot create the run folder: {error}") from error

    experiment = Experiment(
        arm=args.arm,
        config=config,
        generator=FakeGenerator(),
        judge=judge,
        test_runner=FakeTestRunner(),
        store=store,
        prompts=train_prompts,
        eval_prompts=eval_prompts,
    )
    summary = experiment.run()

    print(f"arm={summary.arm} steps={summary.steps} rollouts={summary.rollouts}")
    print(
        f"mean_reward={summary.mean_reward:.4f} pass_at_1={summary.pass_at_1:.4f} "
        f"run folder: {summary.run_dir}"
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
        if args.command == "publish":
            return publish_command(args)
        raise UsageError(f"unknown command {args.command!r}")
    except (UsageError, ConfigError, MissingAPIKeyError) as error:
        print(f"rlvr: {error}", file=sys.stderr)
        return USAGE_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
