"""Command line interface.

    rlvr run --arm <code-score|hybrid|gated-judge|gated-process> --fake
    rlvr screen --questions <path> [--samples <n>] --output <dir> [--fake]
    rlvr eval-band --band <band> [--count 100] [--seed 2026] [--checkpoint <dir>] [--out <path>]
    rlvr summary <run-dir|runs-root> [...] [--html report.html] [--no-html]
    rlvr dashboard [--runs-root <runs-root>] [--port 8081]
    rlvr import-mlflow --runs-root <runs-root> [--arm <arm>]
    rlvr sync-mlflow <run folder> [--config PATH]
    rlvr saved-view --runs-root <runs-root>
    rlvr publish --report report.html --remote <url-or-path> [--branch gh-pages]

This build runs both the fake path and, on a GPU host with the `train` extra
installed, the shared causal language model generator, Docker runner, and GRPO trainer. The judge
backend is chosen before the run folder is made, so a judge arm with no key
fails at startup. `--resume` continues the latest run folder from its last
checkpoint.

`screen` samples the base model with thinking off and writes a difficulty table.
It takes sixteen samples per question by default and `--samples` overrides
that. It never trains, builds no optimizer, and writes into its own output
folder, not a run folder. A restart skips the question ids already in the
table.

`eval-band` scores one labelled difficulty band from the training split. It
prints one JSON line with pass@1 for that band, and writes the same line to
`--out` when that path is given. The test split is not used, and no run folder
is made.

`summary` reads run folders that already exist. It prints one table per metric
with arms side by side and writes a static, self-contained HTML report.
`publish` force-pushes that report to a Pages branch. It needs a remote the
caller's git can already write to.

`import-mlflow` copies old run folders into the local MLflow store. It walks
`runs/<arm>/<timestamp>/meta.json` when `--arm` is omitted, and replays each
folder's metrics, tables and GPU samples.

`sync-mlflow` replays one run folder and deletes the runs it had before, so a
folder that was imported twice or copied from another machine keeps one run.
The store is the one `--config` names, else `MLFLOW_TRACKING_URI`, else the
local `mlflow.db` under the runs root.

A run with `[mlflow] bucket` set starts the uploader beside training, tags the
run with `rlvr/checkpoints_uri`, and never stops training for an upload fault.

`saved-view` writes the study saved view to the `rlvr` experiment and prints the
hash route that opens it. A second call replaces the same tag.

Exit codes: 0 success, 2 for a bad command line, unknown or unimplemented arm,
bad config, a missing judge key, a bad run folder, a bad question set, or a
failed publish. Messages for those cases go to stderr without a traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from .batching import START_PROMPTS, BatchSizePolicy
from .config import (
    Config,
    ConfigError,
    MlflowSettings,
    default_config_path,
    load_config,
    mlflow_settings_from_raw,
)
from .control import CONTROL_NAME, ControlWatcher
from .checkpoint import find_latest_checkpoint, read_checkpoint_manifest
from .dashboard_io import _read_json
from .data import (
    DEFAULT_LABELS_PATH,
    TEST_SIZE,
    TRAIN_SIZE,
    DatasetError,
    MBPPQuestion,
    load_dataset,
    load_difficulty_labels,
    load_split,
    prompts_for_questions,
)
from .experiment import (
    EVAL_TEMPERATURE,
    Experiment,
    Judge,
    JudgeResult,
    Prompt,
)
from .evaluation import band_eval_prompts, pass_at_1
from .fake import (
    FakeGenerator,
    FakeJudge,
    FakeTestRunner,
    fake_memory,
)
from .gpu import GPU_ROWS_NAME, GpuSampler
from .judge import JudgeClient, MissingAPIKeyError
from .memory import configure_expandable_segments, read_torch_memory
from .mlflow_import import import_run, run_dirs
from .mlflow_log import open_mirror
from .mlflow_view import build_route, write_view_tag
from .pools import DEFAULT_PROMOTION_BAR, Pools, PoolsError, from_rows
from .publish import PublishError, publish_report
from .reward import (
    ARMS,
    ArmNotImplemented,
    all_tests_passed,
    check_arm_implemented,
    uses_output_judge,
    uses_process_judge,
)
from .resume import (
    ResumeError,
    manifest_count as _manifest_count,
    resume_batch_policy as _resume_batch_policy,
    validate_resume,
)
from .runs import RunStore
from .schedule import Spread
from .screening import DEFAULT_SCREEN_SAMPLES, ROWS_NAME, ScreenError, ScreenStore
from .screening import screen as run_screen
from .stages import BANDS
from .summary import SummaryError, format_summary, load_reports, write_report
from .synthetic import SyntheticError, load_synthetic_questions
from .upload import RunUploader, upload_prefix

__all__ = [
    "build_generator",
    "build_judge",
    "build_parser",
    "build_pools",
    "build_prompts",
    "build_screen_test_runner",
    "build_test_runner",
    "dashboard_argv",
    "eval_band_command",
    "load_synthetic_rows",
    "main",
    "screen_command",
    "spread_share_warning",
    "sync_mlflow_command",
    "sync_mlflow_settings",
    "training_steps",
]

USAGE_ERROR = 2

#: vLLM shares the card with the training model, so the engine gets a fraction
#: of the memory and a context that covers one prompt plus one completion.
VLLM_GPU_MEMORY_UTILISATION = 0.45


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
        "--control",
        type=Path,
        default=None,
        help="control file read between steps (default: <run folder>/control.json)",
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
        default=DEFAULT_SCREEN_SAMPLES,
        metavar="N",
        help=f"samples per question (default: {DEFAULT_SCREEN_SAMPLES})",
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

    eval_band = subparsers.add_parser(
        "eval-band",
        help="score one labelled difficulty band from the training split",
    )
    eval_band.add_argument(
        "--band",
        required=True,
        help=f"one of: {', '.join(BANDS)}",
    )
    eval_band.add_argument(
        "--count",
        type=int,
        default=100,
        metavar="N",
        help="prompts to score (default: 100)",
    )
    eval_band.add_argument(
        "--seed",
        type=int,
        default=2026,
        metavar="N",
        help="subset seed (default: 2026)",
    )
    eval_band.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="DIR",
        help="adapter checkpoint to load (default: the base model)",
    )
    eval_band.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="optional JSON file for the result (default: print only)",
    )
    eval_band.add_argument(
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
        "dashboard", help="serve the MLflow tracking UI"
    )
    dashboard.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="runs root that holds the MLflow store (default: ./runs)",
    )
    dashboard.add_argument(
        "--port",
        type=int,
        default=8081,
        help="port to serve on (default: 8081)",
    )

    import_mlflow = subparsers.add_parser(
        "import-mlflow", help="copy old run folders into the MLflow store"
    )
    import_mlflow.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="folder that holds run folders (default: ./runs)",
    )
    import_mlflow.add_argument(
        "--arm",
        default=None,
        help="only import this arm (default: every arm under the runs root)",
    )

    sync_mlflow = subparsers.add_parser(
        "sync-mlflow", help="replay one run folder and replace its old run"
    )
    sync_mlflow.add_argument(
        "run_dir",
        type=Path,
        help="run folder, for example runs/code-score/20260101T000000Z",
    )
    sync_mlflow.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config file that names the tracking URI (default: $MLFLOW_TRACKING_URI)",
    )

    saved_view = subparsers.add_parser(
        "saved-view", help="write the study saved view and print its route"
    )
    saved_view.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="runs root that holds the MLflow store (default: ./runs)",
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
    file stops the run before a folder is made, because the synthetic rows are
    part of the fixed training set.
    """
    try:
        return tuple(
            question.as_question() for question in load_synthetic_questions()
        )
    except SyntheticError as error:
        raise UsageError(f"cannot load the synthetic questions: {error}") from error


def build_prompts(config: Config) -> tuple[list[Prompt], list[Prompt]]:
    """Load the fixed split and return the train and test prompts.

    With no measured screen, training uses the 711 MBPP train questions plus the
    50 synthetic very easy questions. With `screen_rows` set, training uses only
    the 711 MBPP train questions, because the screen must cover every trained
    prompt. The synthetic rows were scaffolding for the deleted curriculum
    stage. Evaluation uses the 250 test questions. The split and the prompt
    format come from `data.py`, so every arm and the fake path see the same
    questions. The two lists never share a question, and a synthetic id can
    never collide with an MBPP id. A wrong or empty split stops the run before a
    folder is made.
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

    train_questions = list(split.train)
    if config.screen_rows is None:
        train_questions += list(load_synthetic_rows())
    train_prompts = prompts_for_questions(train_questions)
    eval_prompts = prompts_for_questions(split.test)
    if not train_prompts or not eval_prompts:
        raise UsageError("the split produced an empty train or test prompt list")
    return train_prompts, eval_prompts


def build_pools(config: Config, prompts: Sequence[Prompt]) -> Pools | None:
    """Build the measured pools from the screen rows, or return None.

    `[training] screen_rows` may name the screen output folder or its
    `rows.jsonl` file. An absent key keeps the unmeasured placeholder path,
    where the experiment trains every prompt and the run record says the pool
    was not measured.

    Every row must belong to the supplied training prompts. A row from another
    split would send a prompt id the run cannot train, so it is a config error,
    not a silently ignored row. Every training prompt must also have a row,
    because a partial screen would leave prompts in no pool and train less than
    the operator asked for.
    """
    if config.screen_rows is None:
        return None

    path = Path(config.screen_rows)
    if not path.exists():
        raise UsageError(f"screen_rows path not found: {path}")
    if path.is_dir():
        directory = path
    elif path.name == ROWS_NAME:
        directory = path.parent
    else:
        raise UsageError(
            f"screen_rows must be a screen folder or a {ROWS_NAME} file, "
            f"got {path}"
        )

    try:
        rows = ScreenStore(directory).read_rows()
    except ScreenError as error:
        raise UsageError(f"cannot read screen rows: {error}") from error
    if not rows:
        raise UsageError(f"screen rows are empty: {path}")

    samples = rows[0].samples
    if samples != config.generations_per_prompt:
        raise UsageError(
            f"config: screen rows used {samples} samples per prompt, but "
            f"'training.generations_per_prompt' is "
            f"{config.generations_per_prompt}. A screen must use the training "
            "group size, or the pool labels describe a different rollout count."
        )
    try:
        pools = from_rows(rows, samples=samples, promotion_bar=DEFAULT_PROMOTION_BAR)
    except PoolsError as error:
        raise UsageError(f"cannot build the pools: {error}") from error

    known = {prompt.prompt_id for prompt in prompts}
    measured = set(pools.train) | set(pools.retired) | set(pools.parked)
    unknown = sorted(measured - known)
    if unknown:
        raise UsageError(
            f"screen rows name {len(unknown)} prompt ids that are not in the "
            f"training split, for example {unknown[0]!r}"
        )
    missing = sorted(known - measured)
    if missing:
        shown = ", ".join(repr(prompt_id) for prompt_id in missing[:3])
        raise UsageError(
            f"screen rows cover {len(measured)} of {len(known)} training "
            f"prompts, {len(missing)} missing, for example {shown}"
        )
    return pools


def spread_share_warning(config: Config, pools: Pools | None) -> str | None:
    """A warning when the parked pool cannot fill the planned parked block.

    The spread policy plans half of the step for parked prompts, with a minimum
    of four and at most one fewer than the step. A smaller parked pool leaves
    the block short, so the step spends less on parked work than planned. This
    is a warning, not a failure.
    """
    if config.pool_policy != "spread" or pools is None:
        return None
    group = config.generations_per_prompt
    prompts_per_step = max(1, config.rollouts_per_step // group)
    planned = Spread(prompts_per_step=prompts_per_step).parked_count()
    parked = len(pools.parked)
    if parked >= planned:
        return None
    return (
        f"rlvr: warning: the parked pool holds {parked} prompts, below the "
        f"planned parked block of {planned}. The planned parked share is "
        f"{planned}/{prompts_per_step} of each step."
    )


def training_steps(
    config: Config, prompts: Sequence[Prompt], pools: Pools | None
) -> int:
    """The step count for one run over `config.epochs` epochs.

    With a pool policy the step size comes from the rollout budget, so an epoch
    is one pass over the measured train pool at that size. Without one the old
    path applies: an epoch is one pass at the starting batch size. Both round
    up, so a short tail still gets a step.
    """
    if config.pool_policy is not None and pools is not None:
        available = len(pools.train)
        per_step = max(1, config.rollouts_per_step // config.generations_per_prompt)
        batches = -(-available // per_step)
    else:
        batches = -(-len(prompts) // START_PROMPTS)
    return config.epochs * batches


def build_test_runner(config: Config, *, fake: bool):
    """The Docker runner over the full split and the synthetic rows, or the fake.

    The real runner holds every question because training rollouts run the
    train questions and the synthetic easy questions, and evaluation runs the
    held-out test questions. The fake runner ignores questions entirely.
    """
    if fake:
        return FakeTestRunner()
    try:
        split = load_split(seed=config.split_seed)
    except DatasetError as error:
        raise UsageError(f"cannot load the MBPP split: {error}") from error
    questions = list(split.train) + list(split.test) + list(load_synthetic_rows())
    from .docker_runner import DockerTestRunner

    return DockerTestRunner(questions, workers=config.test_workers)


def _qwen_generator(config: Config, *, trainable: bool):
    """Build the Hugging Face generator, with one startup message on a miss."""
    from .qwen import QwenGenerator

    try:
        return QwenGenerator(
            model_name=config.model_name,
            revision=config.model_revision,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_targets=config.lora_targets,
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
            "install with pip install -e '.[vllm]' or pass --fake. "
            f"({error})"
        ) from error
    except ValueError as error:
        raise UsageError(f"cannot configure the model: {error}") from error


def build_generator(config: Config, *, fake: bool, trainable: bool = True):
    """The real generator, or the fake generator.

    Every real generator samples through vLLM. `trainable` false loads the base
    model with no adapter and no optimizer, and the engine samples those base
    weights. The difficulty screen uses that. Training keeps the default, so
    the adapter is written and handed to the engine after each step.
    """
    if fake:
        return FakeGenerator()
    from .model_profiles import validate_inference_engine

    try:
        validate_inference_engine(config.model_name)
    except ValueError as error:
        raise UsageError(str(error)) from error

    from .vllm_backend import VLLMEngine, VLLMGenerator

    # Build the engine first, so a missing vllm package fails before the
    # training model is loaded.
    try:
        engine = VLLMEngine(
            model_name=config.model_name,
            revision=config.model_revision,
            gpu_memory_utilisation=VLLM_GPU_MEMORY_UTILISATION,
            max_model_len=config.max_prompt_tokens + config.max_completion_tokens,
            max_lora_rank=config.lora_rank,
        )
    except ImportError as error:
        raise UsageError(
            "the rollout engine needs the vllm package. install it with "
            f"pip install -e '.[vllm]'. ({error})"
        ) from error
    try:
        trainer = _qwen_generator(config, trainable=trainable)
        return VLLMGenerator(trainer=trainer, engine=engine)
    except BaseException:
        try:
            engine.close()
        except Exception:
            pass  # Preserve the model startup failure if cleanup also fails.
        raise


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


def start_gpu_sampler(sampler: GpuSampler) -> GpuSampler | None:
    """Start the sampler. A thread that cannot start leaves training running.

    A host with no free threads, or a broken thread implementation, must not
    stop a run. The caller gets None and skips the stop.
    """
    try:
        sampler.start()
    except Exception as error:  # noqa: BLE001 - the sampler must not stop training
        print(f"gpu sampler did not start: {error}", file=sys.stderr)
        return None
    return sampler


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
    # With screen rows the pool is measured and the schedule can draw parked
    # prompts. Without them the run keeps the placeholder path: every prompt is
    # train and the run record says the pool was not measured.
    pools = build_pools(config, train_prompts)
    warning = spread_share_warning(config, pools)
    if warning is not None:
        print(warning, file=sys.stderr)
    labels_path = None if args.fake else DEFAULT_LABELS_PATH
    if config.eval_final_questions is not None:
        eval_prompts = eval_prompts[: config.eval_final_questions]

    # Real runs sweep the train questions once per epoch. A schedule uses the
    # rollout budget as the step size, so an epoch is one pass over the measured
    # train pool. Without a schedule the step size is the starting batch. The
    # fake path keeps the small `[fake] steps` count.
    if not args.fake:
        config = replace(
            config, steps=training_steps(config, train_prompts, pools)
        )

    resume_step = 0
    resume_prompt_cursor = 0
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
        validate_resume(store.meta_path, config, arm=args.arm, fake=args.fake)
        manifest = read_checkpoint_manifest(checkpoint.directory)
        if manifest["arm"] != args.arm:
            raise UsageError(f"checkpoint arm mismatch: {manifest['arm']!r}")
        resume_prompt_cursor = _manifest_count(manifest, "prompt_cursor", 0)
        batch_policy = _resume_batch_policy(config, manifest)
        resume_step = checkpoint.step

    # Build the generator before the run folder is created, so a failed model
    # load leaves no empty folder behind. Resume loads the checkpoint afterwards.
    generator = build_generator(config, fake=args.fake)
    if args.resume:
        try:
            generator.load_checkpoint(checkpoint.directory)
        except Exception as error:
            raise UsageError(f"cannot load checkpoint {checkpoint.directory}: {error}") from error
        # Loading can fail for an incomplete adapter or incompatible model.
        # Keep the recorded run intact until weights and optimizer are loaded.
        store.truncate_to_step(checkpoint.step)
        prior_evaluations = store.read_evaluations()

    if not args.resume:
        try:
            store = RunStore(args.runs_root, args.arm)
        except OSError as error:
            raise UsageError(f"cannot create the run folder: {error}") from error

    # The mirror copies each run write into the MLflow store. A disabled or
    # broken store hands back a null mirror, so training never depends on it.
    # On resume the store is attached after the truncation, so the rebuild
    # replays the kept records.
    mirror = open_mirror(
        args.runs_root,
        store.run_dir,
        config.mlflow.enabled,
        tracking_uri=config.mlflow.tracking_uri,
        artifact_root=config.mlflow.artifact_root,
        checkpoints_uri=(
            upload_prefix(config.mlflow.bucket, args.arm, store.timestamp)
            if config.mlflow.bucket
            else None
        ),
    )
    store.attach_mirror(mirror)
    if args.resume:
        mirror.rebuild()

    test_runner = build_test_runner(config, fake=args.fake)

    control_path = args.control if args.control is not None else store.run_dir / CONTROL_NAME
    control = ControlWatcher(control_path)
    try:
        control.ensure_file()
    except OSError as error:
        raise UsageError(f"cannot create the control file: {error}") from error

    experiment = Experiment(
        arm=args.arm,
        config=config,
        generator=generator,
        judge=judge,
        test_runner=test_runner,
        store=store,
        prompts=train_prompts,
        pools=pools,
        labels_path=labels_path,
        eval_prompts=eval_prompts,
        resume_step=resume_step,
        prior_evaluations=prior_evaluations,
        fake=args.fake,
        memory=fake_memory if args.fake else read_torch_memory,
        resume_prompt_cursor=resume_prompt_cursor,
        batch_policy=batch_policy,
        control=control,
    )
    # The GPU sampler runs beside training. It writes gpu.jsonl and copies each
    # row to the child run. A failed read is dropped, so a host without
    # nvidia-smi still trains.
    #
    # The child run opens at the first write_meta. A row sampled before that is
    # already on disk, so the copy waits for the next tick instead of writing a
    # startup error.
    def sample_gpu(row: Mapping[str, Any]) -> None:
        if mirror.gpu_run_id:
            mirror.gpu_sample(row)

    sampler = start_gpu_sampler(
        GpuSampler(
            store.run_dir / GPU_ROWS_NAME,
            sample_gpu,
            interval=config.mlflow.gpu_sample_seconds,
        )
    )

    # The uploader copies checkpoints and the run folder to the bucket beside
    # training. It starts here, one line above the try, so every exit from the
    # run reaches its close. A missing boto3, a bad destination or a dead
    # network leaves the run folder intact and the run keeps going, so a failed
    # start is a line on stderr and no uploader.
    uploader: RunUploader | None = None
    if config.mlflow.bucket:
        try:
            uploader = RunUploader(
                store.run_dir,
                args.arm,
                store.timestamp,
                config.mlflow.bucket,
                concurrency=config.mlflow.upload_concurrency,
            )
            uploader.start()
        except Exception as error:  # noqa: BLE001 - the uploader must not stop training
            print(f"uploader did not start: {error}", file=sys.stderr)
            uploader = None

    failed = False
    try:
        summary = experiment.run()
    except BaseException:
        failed = True
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        if uploader is not None:
            try:
                uploader.close()
            except Exception as error:  # noqa: BLE001 - the uploader must not stop training
                print(f"uploader did not stop cleanly: {error}", file=sys.stderr)
        mirror.close(status="FAILED" if failed else "FINISHED")

    print(f"arm={summary.arm} steps={summary.steps} rollouts={summary.rollouts}")
    print(
        f"mean_reward={summary.mean_reward:.4f} pass_at_1={summary.pass_at_1:.4f} "
        f"stop_reason={summary.stop_reason} stage={summary.stage or '-'} "
        f"run folder: {summary.run_dir}"
    )
    return 0


def screen_command(args: argparse.Namespace) -> int:
    """Sample the base model and write the difficulty table.

    The sample count defaults to sixteen and `--samples` overrides it.
    """
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


def eval_band_command(args: argparse.Namespace) -> int:
    """Score one labelled difficulty band from the training split."""
    if args.count < 1:
        raise UsageError(f"--count must be at least 1, got {args.count}")

    config = load_config(args.config if args.config is not None else default_config_path())
    try:
        split = load_split(seed=config.split_seed)
        labels = load_difficulty_labels(split)
    except DatasetError as error:
        raise UsageError(f"cannot load the labelled training split: {error}") from error

    try:
        prompts = band_eval_prompts(
            prompts_for_questions(split.train),
            labels,
            args.band,
            args.count,
            args.seed,
        )
    except ValueError as error:
        raise UsageError(str(error)) from error

    # A checkpoint needs an adapter, so that path builds a trainable trainer.
    # Without one the base weights are sampled and no checkpoint is written.
    generator = build_generator(
        config, fake=False, trainable=args.checkpoint is not None
    )
    if args.checkpoint is not None:
        try:
            generator.load_checkpoint(args.checkpoint)
        except Exception as error:
            raise UsageError(
                f"cannot load checkpoint {args.checkpoint}: {error}"
            ) from error
    test_runner = build_test_runner(config, fake=False)

    started = perf_counter()
    passed: list[bool] = []
    offset = 0
    while offset < len(prompts):
        chunk = list(prompts[offset : offset + config.eval_batch_size])
        groups = generator.generate_batch(chunk, n=1, temperature=EVAL_TEMPERATURE)
        if len(groups) != len(chunk) or any(len(group) != 1 for group in groups):
            raise UsageError("evaluation needs exactly one generation per prompt")
        for prompt, group in zip(chunk, groups):
            outcome = test_runner.run(prompt, group[0].code)
            passed.append(
                all_tests_passed(outcome.tests_passed, outcome.tests_total)
            )
        offset += len(chunk)
    seconds = perf_counter() - started

    result = {
        "model": config.model_name,
        "band": args.band,
        "pass_at_1": pass_at_1(passed),
        "count": len(passed),
        "passed": sum(1 for value in passed if value),
        "seconds": seconds,
    }
    line = json.dumps(result)
    print(line)
    if args.out is not None:
        try:
            args.out.write_text(line + "\n", encoding="utf-8")
        except OSError as error:
            raise UsageError(f"cannot write {args.out}: {error}") from error
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


def import_mlflow_command(args: argparse.Namespace) -> int:
    """Copy old run folders into the local MLflow store."""
    runs_root = args.runs_root
    if not runs_root.is_dir():
        raise UsageError(f"runs root not found: {runs_root}")
    try:
        folders = list(run_dirs(runs_root, args.arm))
    except OSError as error:
        raise UsageError(f"cannot read the runs root: {error}") from error
    if not folders:
        print(f"no run folders under {runs_root}")
        return 0
    for run_dir in folders:
        try:
            run_id = import_run(runs_root, run_dir)
        except Exception as error:  # noqa: BLE001 - report the folder and stop
            raise UsageError(f"cannot import {run_dir}: {error}") from error
        label = f"{run_dir.parent.name}/{run_dir.name}"
        if run_id is None:
            print(f"skipped {label} (already imported)")
        else:
            print(f"imported {label} -> {run_id}")
    return 0


def sync_mlflow_settings(
    config_path: Path | None, meta: Mapping[str, Any]
) -> MlflowSettings:
    """The `[mlflow]` settings for `sync-mlflow`.

    A `--config` file wins. Without one the `config` snapshot in `meta.json` is
    read through the same `[mlflow]` parser the run used, so a copied folder
    points back at the store and the bucket it came from. When neither names a
    tracking URI, `MLFLOW_TRACKING_URI` is used, and with none of those the
    mirror writes to `mlflow.db` under the runs root.
    """
    if config_path is not None:
        settings = load_config(config_path).mlflow
    else:
        snapshot = meta.get("config")
        settings = (
            mlflow_settings_from_raw(dict(snapshot))
            if isinstance(snapshot, Mapping)
            else MlflowSettings()
        )
    if settings.tracking_uri:
        return settings
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not env_uri:
        return settings
    return replace(settings, tracking_uri=env_uri)


def sync_mlflow_command(args: argparse.Namespace) -> int:
    """Replay one run folder and replace the runs it had in the store."""
    run_dir = Path(args.run_dir)
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise UsageError(f"not a run folder (no meta.json): {run_dir}")
    meta = _read_json(meta_path)
    runs_root = run_dir.parent.parent
    settings = sync_mlflow_settings(args.config, meta)
    # The same arm and timestamp source the mirror's own `start` uses.
    arm = meta.get("arm")
    if not isinstance(arm, str) or not arm:
        arm = run_dir.parent.name
    timestamp = meta.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        timestamp = run_dir.name
    checkpoints_uri = (
        upload_prefix(settings.bucket, arm, timestamp) if settings.bucket else None
    )
    try:
        run_id = import_run(
            runs_root,
            run_dir,
            rebuild=True,
            tracking_uri=settings.tracking_uri,
            checkpoints_uri=checkpoints_uri,
        )
    except Exception as error:  # noqa: BLE001 - report the folder and stop
        raise UsageError(f"cannot sync {run_dir}: {error}") from error
    if run_id is None:
        raise UsageError(f"cannot sync {run_dir}: the store kept no run id")
    print(f"synced {run_dir.parent.name}/{run_dir.name} -> {run_id}")
    return 0


def saved_view_command(args: argparse.Namespace) -> int:
    """Write the saved view tag and print the route that opens it."""
    try:
        experiment_id = write_view_tag(args.runs_root)
    except Exception as error:  # noqa: BLE001 - report the store fault and stop
        raise UsageError(f"cannot write the saved view: {error}") from error
    print(build_route(experiment_id))
    return 0


def dashboard_argv(runs_root: Path, port: int) -> list[str]:
    """The command that starts the MLflow server.

    The interpreter that runs `rlvr` also runs MLflow as a module. The `mlflow`
    console script is not on PATH on a bare host, so the module form is the one
    that works from `.venv/bin/rlvr`.
    """
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{runs_root / 'mlflow.db'}",
        "--artifacts-destination",
        str(runs_root / "mlartifacts"),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--allowed-hosts",
        os.environ.get("RLVR_ALLOWED_HOSTS", "*"),
    ]


def dashboard_command(args: argparse.Namespace) -> int:
    """Start the MLflow server on the runs root."""
    if not 0 < args.port <= 65535:
        raise UsageError(f"invalid port {args.port}")
    runs_root = args.runs_root
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "mlartifacts").mkdir(parents=True, exist_ok=True)
    command = dashboard_argv(runs_root, args.port)
    print(f"dashboard: http://0.0.0.0:{args.port}  (Ctrl+C to stop)")
    try:
        os.execv(command[0], command)
    except OSError as error:
        raise UsageError(f"cannot start mlflow server: {error}") from error
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
        if args.command == "eval-band":
            return eval_band_command(args)
        if args.command == "summary":
            return summary_command(args)
        if args.command == "dashboard":
            return dashboard_command(args)
        if args.command == "import-mlflow":
            return import_mlflow_command(args)
        if args.command == "sync-mlflow":
            return sync_mlflow_command(args)
        if args.command == "saved-view":
            return saved_view_command(args)
        if args.command == "publish":
            return publish_command(args)
        raise UsageError(f"unknown command {args.command!r}")
    except (UsageError, ConfigError, ResumeError, MissingAPIKeyError) as error:
        print(f"rlvr: {error}", file=sys.stderr)
        return USAGE_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
