"""Command line interface.

    rlvr run --arm <code-score|hybrid|gated-judge|gated-process> --fake

Task 1 implements the fake path only. Real backends arrive in later tasks and
plug into the same Experiment seam without changing this parser.

Exit codes: 0 success, 2 for a bad command line, unknown arm, or bad config.
Messages for those cases go to stderr without a traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError, default_config_path, load_config
from .experiment import Experiment
from .fake import FakeGenerator, FakeJudge, FakeTestRunner, fake_prompts
from .reward import ARMS
from .runs import RunStore

__all__ = ["build_parser", "main"]

USAGE_ERROR = 2


class UsageError(ValueError):
    """A command line problem the user can fix."""


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
    return parser


def run_command(args: argparse.Namespace) -> int:
    if args.arm not in ARMS:
        raise UsageError(f"unknown arm {args.arm!r}. valid arms: {', '.join(ARMS)}")
    if not args.fake:
        raise UsageError(
            f"real backends for arm {args.arm!r} are not implemented yet, add --fake"
        )

    config = load_config(args.config if args.config is not None else default_config_path())
    try:
        store = RunStore(args.runs_root, args.arm)
    except OSError as error:
        raise UsageError(f"cannot create the run folder: {error}") from error

    experiment = Experiment(
        arm=args.arm,
        config=config,
        generator=FakeGenerator(),
        judge=FakeJudge(),
        test_runner=FakeTestRunner(),
        store=store,
        prompts=fake_prompts(config.prompts_per_step),
    )
    summary = experiment.run()

    print(f"arm={summary.arm} steps={summary.steps} rollouts={summary.rollouts}")
    print(
        f"mean_reward={summary.mean_reward:.4f} pass_at_1={summary.pass_at_1:.4f} "
        f"run folder: {summary.run_dir}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_command(args)
        raise UsageError(f"unknown command {args.command!r}")
    except (UsageError, ConfigError) as error:
        print(f"rlvr: {error}", file=sys.stderr)
        return USAGE_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
