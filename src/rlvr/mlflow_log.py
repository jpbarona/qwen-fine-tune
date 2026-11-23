"""A one way mirror from a run folder into a local MLflow store.

The run folder stays the record. Every file is written before this module is
called, so a store that is missing, locked or broken loses a copy and nothing
else. Training keeps running.

The store is one SQLite file under the runs root, with one experiment named
`rlvr`. A run gets a parent run for the step series and a nested child run for
the GPU samples. The two run ids and the table append counts are written to
`mlflow.json` in the run folder, so `meta.json` stays immutable.

Every public call catches its own exceptions and appends one line to
`mlflow-errors.log` in the run folder. A failed mirror never raises into the
training loop.

`eval` pushes the run tables and the `eval.json` artifact. `rebuild` deletes
the parent and child runs and replays the local files on resume. `close_dead`
marks the stale runs of one arm as failed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import tempfile
import warnings
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .dashboard_io import _read_json, _read_jsonl
from .gpu import GPU_ROWS_NAME
from .runread import (
    META_NAME,
    METRICS_NAME,
    SERIES_FIELDS,
    collect_dashboard,
    target_steps,
)

__all__ = ["Mirror", "MlflowMirror", "NullMirror", "open_mirror"]

EXPERIMENT_NAME = "rlvr"
DATABASE_NAME = "mlflow.db"
ARTIFACTS_NAME = "mlartifacts"
IDS_NAME = "mlflow.json"
ERRORS_NAME = "mlflow-errors.log"
EVAL_NAME = "eval.json"
LABELS_NAME = "labels.jsonl"
CONTROL_EVENTS_NAME = "control-events.jsonl"

#: One questions and one distributions file per evaluation, under these
#: folders. The step and the evaluation kind are in the name, so a later eval
#: never replaces an earlier one and two evals at one step both survive.
QUESTIONS_DIR = "questions"
DISTRIBUTIONS_DIR = "distributions"

#: Tables that grow over a run. Each eval appends the rows that arrived since
#: the last one.
EVALUATIONS_TABLE = "evaluations.json"
LABELS_TABLE = "labels.json"
DROPPED_TABLE = "dropped.json"
CONTROL_EVENTS_TABLE = "control-events.json"

#: The append tables, in the order the counts are written to `mlflow.json`.
APPENDED_TABLES: tuple[str, ...] = (
    EVALUATIONS_TABLE,
    LABELS_TABLE,
    DROPPED_TABLE,
    CONTROL_EVENTS_TABLE,
)

#: MLflow 3.18 calls two pandas APIs that pandas 3 deprecates. Nothing here
#: can fix that, so the two messages are dropped around a table write instead
#: of printing once per eval in the training log.
TABLE_WARNINGS: tuple[str, ...] = (
    "For backward compatibility, 'str' dtypes are included by select_dtypes",
    "The 'convert_dates' keyword in read_json is deprecated",
)

#: MLflow reads this setting when system metrics logging starts, so setting it
#: here covers the run the mirror opens next. A caller that set its own value
#: keeps it.
SYSTEM_METRICS_INTERVAL_ENV = "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"
SYSTEM_METRICS_INTERVAL = "1"

#: The tag that records when the mirror last copied a step. A run with no tag
#: is left alone by `close_dead`, because an old run may predate the sweep.
LAST_WRITE_TAG = "rlvr/last_write_utc"

#: The name used for the heartbeat tag and the heartbeat metric. The value is
#: a decimal string of seconds since the run started. The spec maps the name
#: to a metric on the parent run and to a tag on both runs.
HEARTBEAT_TAG = "rlvr/heartbeat_seconds"

#: The tag that records how many prompts the run dropped. The spec puts the
#: count beside the dropped table.
DROPPED_COUNT_TAG = "rlvr/dropped_count"

#: The metric names for the pass@1 series. The spec keeps the checkpoint and
#: final scores as separate series so a run can show both.
PASS_AT_1_METRICS: tuple[str, ...] = ("checkpoint", "final")

#: Fields copied from `meta.json` to a tag. The rest of the metadata goes to
#: params, flattened with dots.
TAG_FIELDS: tuple[tuple[str, str], ...] = (
    ("git_commit", "rlvr/git_commit"),
    ("model_revision", "rlvr/model_revision"),
    ("pool_policy", "rlvr/pool_policy"),
    ("mode", "rlvr/mode"),
)


class Mirror(Protocol):
    """The seam the training store writes through. A null version does nothing."""

    run_id: str
    gpu_run_id: str

    def start(self, meta: Mapping[str, Any]) -> None: ...

    def step(self, row: Mapping[str, Any]) -> None: ...

    def eval(self, payload: Mapping[str, Any]) -> None: ...

    def labels(self) -> None: ...

    def gpu_sample(self, row: Mapping[str, Any]) -> None: ...

    def rebuild(self) -> None: ...

    def close_dead(self, arm: str, older_than_seconds: float = 600.0) -> None: ...

    def close(self, status: str = "FINISHED") -> None: ...


def _guarded(method: Callable[..., Any]) -> Callable[..., Any]:
    """Turn an exception into one line in the run folder's error log."""

    @wraps(method)
    def wrapper(self: "MlflowMirror", *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except Exception as error:  # noqa: BLE001 - the mirror must not stop training
            self._write_error(error)
            return None

    return wrapper


def _number(value: Any) -> float | None:
    """A finite float, or None. Booleans and text are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _start_seconds(info: Any) -> float:
    """The run's start time as Unix seconds. A missing time falls back to now."""
    started = getattr(info, "start_time", None)
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        return float(started) / 1000.0
    return datetime.now(timezone.utc).timestamp()


def _age_seconds(stamp: str, now: datetime) -> float | None:
    """Seconds since an ISO 8601 stamp, or None when the stamp is not one.

    A stamp with no timezone is read as UTC. A stamp that cannot be parsed is
    treated as unknown, so the run is left alone rather than guessed at.
    """
    try:
        written = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if written.tzinfo is None:
        written = written.replace(tzinfo=timezone.utc)
    return (now - written).total_seconds()


def _flatten(prefix: str, value: Any, into: dict[str, Any]) -> None:
    """Flatten nested dicts into dotted param names. Lists and nulls are skipped."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}", child, into)
        return
    if value is None or isinstance(value, (list, tuple, set)):
        return
    if isinstance(value, (str, int, float, bool)):
        into[prefix] = value


def _params(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The scalar leaves of the metadata, as MLflow params.

    `versions` is a map of package names to versions. It says little about a
    run and would swamp the param list, so it is left out.
    """
    params: dict[str, Any] = {}
    for key, value in meta.items():
        if key == "versions":
            continue
        _flatten(str(key), value, params)
    return params


def _table(rows: list[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Rows as the column map `log_table` takes. An empty table is empty."""
    if not rows:
        return {}
    columns = sorted({name for row in rows for name in row})
    return {name: [row.get(name) for row in rows] for name in columns}


def _payload_step(payload: Mapping[str, Any]) -> int:
    """The step of the newest evaluation in an `eval.json` payload, or zero."""
    step = payload.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    evaluations = payload.get("evaluations")
    if isinstance(evaluations, list):
        for point in reversed(evaluations):
            if not isinstance(point, Mapping):
                continue
            value = point.get("step")
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return 0


def _payload_kind(payload: Mapping[str, Any]) -> str:
    """The kind of the newest evaluation in an `eval.json` payload.

    A payload carries the kind at the top level in some writers and on each
    evaluation point in others. The top level wins, then the newest point.
    """
    kind = payload.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    evaluations = payload.get("evaluations")
    if isinstance(evaluations, list):
        for point in reversed(evaluations):
            if not isinstance(point, Mapping):
                continue
            value = point.get("kind")
            if isinstance(value, str) and value:
                return value
    return "unknown"


def _slug(name: str) -> str:
    """A metric name segment: lower case, letters, digits and `_` only."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


def _evaluation_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The evaluation points of `eval.json`, every field kept."""
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        return []
    return [dict(point) for point in evaluations if isinstance(point, Mapping)]


def _load_table_counts(run_dir: Path) -> dict[str, int]:
    """The append counts an earlier mirror wrote to `mlflow.json`, or zeros."""
    payload = _read_json(run_dir / IDS_NAME)
    tables = payload.get("tables")
    tables = tables if isinstance(tables, Mapping) else {}
    counts: dict[str, int] = {}
    for name in APPENDED_TABLES:
        value = tables.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            counts[name] = value
        else:
            counts[name] = 0
    return counts


def _distribution_rows(distributions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The two histograms side by side, one row per bin index."""
    reward = distributions.get("reward")
    reward = reward if isinstance(reward, Mapping) else {}
    tests = distributions.get("tests_passed")
    tests = tests if isinstance(tests, Mapping) else {}
    reward_bins = reward.get("bins") or []
    reward_counts = reward.get("counts") or []
    test_bins = tests.get("bins") or []
    test_counts = tests.get("counts") or []
    length = max(
        len(reward_bins), len(reward_counts), len(test_bins), len(test_counts)
    )

    def at(values: list[Any], index: int) -> Any:
        return values[index] if index < len(values) else None

    return [
        {
            "reward_bin": at(reward_bins, index),
            "reward_count": at(reward_counts, index),
            "tests_passed_bin": at(test_bins, index),
            "tests_passed_count": at(test_counts, index),
        }
        for index in range(length)
    ]


def _dropped_rows(dropped: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per dropped prompt, with the step it was dropped."""
    ids = dropped.get("ids") or []
    steps = dropped.get("steps")
    steps = steps if isinstance(steps, Mapping) else {}
    return [
        {"prompt_id": prompt_id, "dropped_step": steps.get(prompt_id)}
        for prompt_id in ids
        if isinstance(prompt_id, str)
    ]


def _write_error(run_dir: Path, error: BaseException) -> None:
    """Append one line to the error log, and never raise."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{stamp} {type(error).__name__}: {error}\n"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / ERRORS_NAME).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    except Exception:  # noqa: BLE001 - the log must not raise into training
        pass


def _write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file in the same folder."""
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class NullMirror:
    """The mirror used when tracking is off or MLflow is not installed."""

    run_id = ""
    gpu_run_id = ""

    def start(self, meta: Mapping[str, Any]) -> None:
        pass

    def step(self, row: Mapping[str, Any]) -> None:
        pass

    def eval(self, payload: Mapping[str, Any]) -> None:
        pass

    def labels(self) -> None:
        pass

    def gpu_sample(self, row: Mapping[str, Any]) -> None:
        pass

    def rebuild(self) -> None:
        pass

    def close_dead(self, arm: str, older_than_seconds: float = 600.0) -> None:
        pass

    def close(self, status: str = "FINISHED") -> None:
        pass


class MlflowMirror:
    """Copies one run folder into the local MLflow store.

    Construction sets the tracking URI and makes sure the experiment exists.
    `start` opens the parent and the GPU child run. `step` copies the newest
    value of every series. `close` terminates both runs.
    """

    def __init__(self, runs_root: str | Path, run_dir: str | Path) -> None:
        self.runs_root = Path(runs_root)
        self.run_dir = Path(run_dir)
        self.run_id = ""
        self.gpu_run_id = ""
        self.experiment_id = ""
        self._gpu_step = 0
        # The heartbeat is measured from the run's start time. A mirror that
        # has not opened a run yet falls back to its construction time.
        self._started_at = datetime.now(timezone.utc).timestamp()
        self._gpu_started_at = self._started_at
        self._mlflow: Any = None
        self._client: Any = None
        self._question_steps: set[tuple[int, str]] = set()
        self._distribution_steps: set[tuple[int, str]] = set()
        self._stage_steps: set[tuple[int, str]] = set()
        self._band_steps: set[tuple[int, str]] = set()
        self._pass_steps: set[tuple[int, str]] = set()
        # A mirror that found an existing `mlflow.json` only owns the table
        # counts, not the run ids an earlier mirror wrote.
        self._owns_ids = False
        # A mirror rebuilt over the same run folder picks up the counts from
        # the last one, so a restart does not append the same rows twice.
        self._logged_rows: dict[str, int] = _load_table_counts(self.run_dir)
        try:
            self._connect()
        except Exception as error:  # noqa: BLE001 - the mirror must not stop training
            self._write_error(error)

    def _connect(self) -> None:
        """Point MLflow at the runs root and make sure the experiment exists."""
        os.environ.setdefault(SYSTEM_METRICS_INTERVAL_ENV, SYSTEM_METRICS_INTERVAL)
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(f"sqlite:///{self.runs_root / DATABASE_NAME}")
        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = client.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=str(self.runs_root / ARTIFACTS_NAME),
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(EXPERIMENT_NAME)
        self._mlflow = mlflow
        self._client = client
        self.experiment_id = experiment_id

    def _write_error(self, error: BaseException) -> None:
        _write_error(self.run_dir, error)

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("MLflow is not available")
        return self._client

    def _run_name(self, meta: Mapping[str, Any]) -> tuple[str, str, str]:
        arm = _text(meta.get("arm")) or self.run_dir.parent.name
        timestamp = _text(meta.get("timestamp")) or self.run_dir.name
        return arm, timestamp, f"{arm}-{timestamp}"

    @_guarded
    def start(self, meta: Mapping[str, Any]) -> None:
        """Open the parent run, write its metadata, and open the GPU child run."""
        client = self._require_client()
        arm, timestamp, name = self._run_name(meta)

        tags: dict[str, str] = {
            "rlvr/arm": arm,
            "rlvr/run_name": name,
            "rlvr/timestamp": timestamp,
            "rlvr/run_dir": str(self.run_dir.resolve()),
        }
        for source, tag in TAG_FIELDS:
            value = _text(meta.get(source))
            if value is not None:
                tags[tag] = value

        # A crashed run of this arm may still be marked RUNNING. Sweep it
        # before the new run opens, so the old one does not stay green.
        self.close_dead(arm)

        options: dict[str, Any] = {}
        if importlib.util.find_spec("psutil") is not None:
            options["log_system_metrics"] = True
        self._mlflow.set_experiment(EXPERIMENT_NAME)
        active = self._mlflow.start_run(run_name=name, **options)
        self.run_id = active.info.run_id
        self._started_at = _start_seconds(active.info)

        self._mlflow.set_tags(tags)
        params = _params(meta)
        target = target_steps(meta)
        if target is not None:
            params["max_steps"] = target
        if params:
            self._mlflow.log_params(params)

        # A new run has no tables yet, so the counts start at zero even when
        # the folder carries counts from an earlier run. The per run sets and
        # the GPU sample counter start fresh too, so a rebuild does not skip
        # a table or step the child run past its first sample.
        self._logged_rows = {name: 0 for name in APPENDED_TABLES}
        self._gpu_step = 0
        self._question_steps.clear()
        self._distribution_steps.clear()
        self._stage_steps.clear()
        self._band_steps.clear()
        self._pass_steps.clear()

        child = client.create_run(
            self.experiment_id,
            run_name=f"{name}-gpu",
            tags={
                "mlflow.parentRunId": self.run_id,
                "rlvr/arm": arm,
                "rlvr/role": "gpu",
            },
        )
        self.gpu_run_id = child.info.run_id
        self._gpu_started_at = _start_seconds(child.info)
        self._owns_ids = True
        # A run that crashes before its first step must still carry a last
        # write and a heartbeat, so the sweep can find it.
        self._write_heartbeat(datetime.now(timezone.utc))
        self._write_ids()

    @_guarded
    def step(self, row: Mapping[str, Any]) -> None:
        """Copy the series values of one metrics row. Nulls are skipped.

        The caller passes the row it just wrote, so the mirror never reads the
        run folder on a training step. The heartbeat tags are refreshed on the
        parent and the child run, so a crashed run leaves an old last write
        behind. The parent run also gets the heartbeat as a metric, because the
        spec maps the name to a metric. The stage and band series move with the
        evaluation, because they are not in a metrics row.
        """
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        index = row.get("step")
        if isinstance(index, bool) or not isinstance(index, int):
            return
        heartbeat = self._write_heartbeat(datetime.now(timezone.utc))
        self._require_client().log_metric(
            self.run_id, HEARTBEAT_TAG, heartbeat, step=index
        )
        self._log_row(row, index)

    def _write_heartbeat(self, now: datetime) -> float:
        """Refresh the last write and heartbeat tags on the parent and child.

        The last write is an ISO 8601 UTC string. The heartbeat is a decimal
        string of seconds since the run started. The child run gets its own
        heartbeat, measured from the child's start time. The parent heartbeat
        is returned so the caller can log it as a metric.
        """
        client = self._require_client()
        stamp = now.isoformat()
        heartbeat = max(0.0, now.timestamp() - self._started_at)
        client.set_tag(self.run_id, LAST_WRITE_TAG, stamp)
        client.set_tag(self.run_id, HEARTBEAT_TAG, f"{heartbeat:.3f}")
        if not self.gpu_run_id:
            return heartbeat
        child_heartbeat = max(0.0, now.timestamp() - self._gpu_started_at)
        client.set_tag(self.gpu_run_id, LAST_WRITE_TAG, stamp)
        client.set_tag(self.gpu_run_id, HEARTBEAT_TAG, f"{child_heartbeat:.3f}")
        return heartbeat

    def _log_stage(self, run: Mapping[str, Any]) -> None:
        """Log every stage progress point not logged yet.

        The dashboard carries the whole stage series. A set keeps a later
        evaluation from logging the same point twice.
        """
        client = self._require_client()
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        stage_progress = run.get("stage_progress") or {}
        for point in stage_progress.get("points") or []:
            if not isinstance(point, Mapping):
                continue
            step = point.get("step")
            band = _text(point.get("band"))
            if isinstance(step, bool) or not isinstance(step, int) or band is None:
                continue
            key = (step, band)
            if key in self._stage_steps:
                continue
            for field in ("solved", "total"):
                value = _number(point.get(field))
                if value is not None:
                    client.log_metric(
                        self.run_id, f"stage/{_slug(band)}/{field}", value, step=step
                    )
            self._stage_steps.add(key)

    def _log_bands(self, step: int, run: Mapping[str, Any]) -> None:
        """Log the band pass rates at `step`.

        The rate is cumulative. A later evaluation logs the same step once, so
        two evaluations at one step do not draw two points on one line.
        """
        client = self._require_client()
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        for band_row in run.get("bands") or []:
            if not isinstance(band_row, Mapping):
                continue
            band = _text(band_row.get("band"))
            value = _number(band_row.get("pass_rate"))
            if band is None or value is None:
                continue
            key = (step, band)
            if key in self._band_steps:
                continue
            client.log_metric(
                self.run_id, f"band/{_slug(band)}/pass_rate", value, step=step
            )
            self._band_steps.add(key)

    def _log_row(self, row: Mapping[str, Any], step: int) -> None:
        """Log the series values of one metrics row at `step`."""
        client = self._require_client()
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        for name in SERIES_FIELDS:
            value = _number(row.get(name))
            if value is not None:
                client.log_metric(self.run_id, name, value, step=step)

    def _log_pass_at_1(self, payload: Mapping[str, Any]) -> None:
        """Log the checkpoint and final pass@1 values from the eval payload.

        A payload carries one point per evaluation. The checkpoint and final
        points become separate metric series. A set keeps a later payload from
        logging the same point twice.
        """
        client = self._require_client()
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        evaluations = payload.get("evaluations")
        if not isinstance(evaluations, list):
            return
        for point in evaluations:
            if not isinstance(point, Mapping):
                continue
            kind = _text(point.get("kind"))
            step = point.get("step")
            value = _number(point.get("pass_at_1"))
            if kind not in PASS_AT_1_METRICS:
                continue
            if isinstance(step, bool) or not isinstance(step, int) or value is None:
                continue
            key = (step, kind)
            if key in self._pass_steps:
                continue
            client.log_metric(self.run_id, f"pass_at_1/{kind}", value, step=step)
            self._pass_steps.add(key)

    def _dashboard_run(self) -> Mapping[str, Any]:
        """The dashboard payload for this run folder, or an empty mapping."""
        dashboard = collect_dashboard([self.run_dir])
        runs = dashboard.get("runs") or []
        return runs[0] if runs else {}

    @_guarded
    def eval(self, payload: Mapping[str, Any]) -> None:
        """Log the run tables and the evaluation metrics in `payload`.

        The stage and band series come from the dashboard payload, read once at
        the evaluation. The pass@1 series and the dropped count come from the
        payload and the dashboard too.
        """
        run = self._dashboard_run()
        step = _payload_step(payload)
        if step == 0:
            series = run.get("series") or {}
            steps = series.get("steps") or []
            if steps and isinstance(steps[-1], int):
                step = steps[-1]
        self._log_stage(run)
        self._log_bands(step, run)
        self._log_pass_at_1(payload)
        self.push_tables(step, run)

    def labels(self) -> None:
        """Placeholder. The label table is pushed with the other tables."""
        pass

    @_guarded
    def push_tables(
        self, step: int, run: Mapping[str, Any] | None = None
    ) -> None:
        """Log the run tables for one evaluation.

        The questions and distributions tables get one file per step and kind,
        so a later eval does not replace an earlier one and two evals at one
        step both survive. The event tables append, and each call logs only the
        rows that arrived since the last call. A table with no rows is not
        written. One failed table is logged and the rest are still pushed. The
        whole `eval.json` is logged as an artifact too. A caller that already
        read the dashboard passes `run`, so the folder is read once.
        """
        client = self._require_client()
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        index = int(step)
        if run is None:
            run = self._dashboard_run()
        payload = _read_json(self.run_dir / EVAL_NAME)
        kind = _payload_kind(payload)
        key = (index, kind)

        questions = (run.get("questions") or {}).get("rows") or []
        if key not in self._question_steps:
            name = f"{QUESTIONS_DIR}/step-{index}-{kind}.json"
            if self._push(name, questions):
                self._question_steps.add(key)

        distributions = run.get("distributions") or {}
        if key not in self._distribution_steps:
            name = f"{DISTRIBUTIONS_DIR}/step-{index}-{kind}.json"
            rows = _distribution_rows(distributions)
            if self._push(name, rows):
                self._distribution_steps.add(key)

        dropped = _dropped_rows(run.get("dropped") or {})
        self._push_appended(EVALUATIONS_TABLE, _evaluation_rows(payload))
        self._push_appended(LABELS_TABLE, _read_jsonl(self.run_dir / LABELS_NAME))
        self._push_appended(DROPPED_TABLE, dropped)
        self._push_appended(
            CONTROL_EVENTS_TABLE, _read_jsonl(self.run_dir / CONTROL_EVENTS_NAME)
        )
        client.set_tag(self.run_id, DROPPED_COUNT_TAG, str(len(dropped)))
        self._log_artifact(self.run_dir / EVAL_NAME)

    def _push(self, artifact_file: str, rows: list[Mapping[str, Any]]) -> bool:
        """Log one table. True when it was written."""
        data = _table(rows)
        if not data:
            return False
        try:
            with warnings.catch_warnings():
                for message in TABLE_WARNINGS:
                    warnings.filterwarnings("ignore", message=message)
                self._mlflow.log_table(
                    data, artifact_file=artifact_file, run_id=self.run_id
                )
        except Exception as error:  # noqa: BLE001 - one table must not stop the rest
            self._write_error(error)
            return False
        return True

    def _log_artifact(self, path: Path) -> None:
        """Log one file as an artifact. A missing file is not an error."""
        if not path.is_file():
            return
        try:
            self._mlflow.log_artifact(str(path), run_id=self.run_id)
        except Exception as error:  # noqa: BLE001 - one artifact must not stop the rest
            self._write_error(error)

    def _push_appended(
        self, artifact_file: str, rows: list[Mapping[str, Any]]
    ) -> None:
        """Append the rows this run has not pushed yet.

        The count is kept in `mlflow.json`, so a mirror rebuilt over the same
        folder skips the rows an earlier one pushed. A file that shrank under
        the table is left alone: the store already has more rows, and pushing
        the whole table again would duplicate them.
        """
        seen = self._logged_rows.get(artifact_file, 0)
        if seen > len(rows):
            # The file shrank under the table. The store already has those
            # rows, so pushing them again would duplicate them. Leave both the
            # file and the count alone.
            return
        fresh = rows[seen:]
        if not fresh:
            return
        if self._push(artifact_file, fresh):
            self._logged_rows[artifact_file] = len(rows)
            self._write_ids()

    @_guarded
    def gpu_sample(self, row: Mapping[str, Any]) -> None:
        """Copy one GPU sample to the child run, stepped by sample index.

        The child run is addressed by id, because the active run is thread
        local and the sampler runs on its own thread.
        """
        client = self._require_client()
        if not self.gpu_run_id:
            raise RuntimeError("the GPU run has not started")
        index = self._gpu_step
        self._gpu_step += 1
        for name, raw in row.items():
            if name == "t":
                continue
            value = _number(raw)
            if value is None:
                continue
            client.log_metric(self.gpu_run_id, name, value, step=index)

    def _runs_for_folder(self) -> list[str]:
        """Every run in the experiment whose `rlvr/run_dir` tag is this folder.

        A crash between `start_run` and the ids write leaves a run that
        `mlflow.json` does not name. The tag finds it. The child runs of each
        matching parent are included too, so a rebuild leaves no GPU run
        behind.
        """
        client = self._require_client()
        escaped = str(self.run_dir.resolve()).replace("'", "\\'")
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.`rlvr/run_dir` = '{escaped}'",
        )
        found: list[str] = []
        for run in runs:
            parent = run.info.run_id
            if parent not in found:
                found.append(parent)
            children = client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{parent}'",
            )
            for child in children:
                if child.info.run_id not in found:
                    found.append(child.info.run_id)
        return found

    @_guarded
    def rebuild(self) -> None:
        """Replace the runs for this folder with a replay of the local files.

        A resumed run deletes every run whose `rlvr/run_dir` tag names this
        folder, then recreates the parent and child runs. The tag finds a run
        left by a crash before `mlflow.json` was written. The local files are
        already truncated to the last checkpoint, so the replay holds no point
        past it. The active run this process owns is ended first, so the new
        run can open.
        """
        meta = _read_json(self.run_dir / META_NAME)
        arm = self._run_name(meta)[0]
        self.close_dead(arm)

        ids = _read_json(self.run_dir / IDS_NAME)
        self.run_id = self.run_id or _text(ids.get("run_id")) or ""
        self.gpu_run_id = self.gpu_run_id or _text(ids.get("gpu_run_id")) or ""

        # End the active run this process owns, so the new run can open. The
        # explicit end runs before close, because a failed child termination
        # would otherwise leave the parent active.
        active = self._mlflow.active_run()
        if active is not None and active.info.run_id == self.run_id:
            self._mlflow.end_run(status="FAILED")
        self.close(status="FAILED")

        client = self._require_client()
        doomed = set(self._runs_for_folder())
        doomed.update(run_id for run_id in (self.run_id, self.gpu_run_id) if run_id)
        for run_id in doomed:
            try:
                client.delete_run(run_id)
            except Exception as error:  # noqa: BLE001 - a missing run is not fatal
                self._write_error(error)

        self.run_id = ""
        self.gpu_run_id = ""
        self.start(meta)
        if not self.run_id:
            return
        self._replay()

    def _replay(self) -> None:
        """Push the local files into the runs `start` just opened."""
        for row in _read_jsonl(self.run_dir / METRICS_NAME):
            self.step(row)
        payload = _read_json(self.run_dir / EVAL_NAME)
        self.eval(payload)
        for row in _read_jsonl(self.run_dir / GPU_ROWS_NAME):
            self.gpu_sample(row)
        self._write_heartbeat(datetime.now(timezone.utc))

    @_guarded
    def close_dead(self, arm: str, older_than_seconds: float = 600.0) -> None:
        """Mark the stale RUNNING runs of one arm as FAILED.

        A run is stale when its `rlvr/last_write_utc` tag is older than the
        threshold. A run with no tag is left alone, because an old run may
        predate the sweep and its age is unknown. The active run this process
        owns is ended through `mlflow`, so the caller can open a new one.
        """
        client = self._require_client()
        now = datetime.now(timezone.utc)
        escaped = arm.replace("'", "\\'")
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=(
                f"tags.`{LAST_WRITE_TAG}` != '' and "
                f"tags.`rlvr/arm` = '{escaped}' and attributes.status = 'RUNNING'"
            ),
        )
        for run in runs:
            stamp = _text(run.data.tags.get(LAST_WRITE_TAG))
            if stamp is None:
                continue
            age = _age_seconds(stamp, now)
            if age is None or age <= older_than_seconds:
                continue
            active = self._mlflow.active_run()
            if active is not None and active.info.run_id == run.info.run_id:
                self._mlflow.end_run(status="FAILED")
            else:
                client.set_terminated(run.info.run_id, "FAILED")

    @_guarded
    def close(self, status: str = "FINISHED") -> None:
        """Terminate the GPU child run and, if it is active, the parent run."""
        client = self._require_client()
        if self.gpu_run_id:
            client.set_terminated(self.gpu_run_id, status)
        if not self.run_id:
            return
        active = self._mlflow.active_run()
        if active is not None and active.info.run_id == self.run_id:
            self._mlflow.end_run(status=status)
        else:
            client.set_terminated(self.run_id, status)

    def _write_ids(self) -> None:
        """Write the run ids and table counts to `mlflow.json`.

        A mirror that opened the run owns both ids and writes them. A mirror
        that found an existing file updates only the table counts, so the ids
        an earlier mirror wrote are left alone. An empty file is filled with
        the ids this mirror has.
        """
        path = self.run_dir / IDS_NAME
        payload = _read_json(path)
        if self._owns_ids or not payload:
            payload["run_id"] = self.run_id
            payload["gpu_run_id"] = self.gpu_run_id
        payload["tables"] = {
            name: self._logged_rows.get(name, 0) for name in APPENDED_TABLES
        }
        _write_atomic(path, json.dumps(payload, indent=2) + "\n")


def open_mirror(runs_root: Path, run_dir: Path, enabled: bool) -> Mirror:
    """The mirror for one run. A disabled or broken mirror does nothing.

    A store that could not be opened at construction leaves one line in
    `mlflow-errors.log` and then a null mirror, so a run never pays for a
    broken backend on every step.
    """
    if not enabled:
        return NullMirror()
    try:
        mirror = MlflowMirror(runs_root, run_dir)
    except Exception as error:  # noqa: BLE001 - the mirror must not stop training
        _write_error(Path(run_dir), error)
        return NullMirror()
    if mirror._client is None:
        return NullMirror()
    return mirror
