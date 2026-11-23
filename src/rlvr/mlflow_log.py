"""A one way mirror from a run folder into a local MLflow store.

The run folder stays the record. Every file is written before this module is
called, so a store that is missing, locked or broken loses a copy and nothing
else. Training keeps running.

The store is one SQLite file under the runs root, with one experiment named
`rlvr`. A run gets a parent run for the step series and a nested child run for
the GPU samples. The two run ids are written to `mlflow.json` in the run
folder, so `meta.json` stays immutable.

Every public call catches its own exceptions and appends one line to
`mlflow-errors.log` in the run folder. A failed mirror never raises into the
training loop.

`eval` pushes the run tables. `labels` and `rebuild` are placeholders until the
resume rebuild and the importer land.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import tempfile
import warnings
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .dashboard_io import _read_json, _read_jsonl
from .runread import collect_dashboard

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
#: folders. The step is in the name, so a later eval never replaces an earlier
#: one.
QUESTIONS_DIR = "questions"
DISTRIBUTIONS_DIR = "distributions"

#: Tables that grow over a run. Each eval appends the rows that arrived since
#: the last one.
EVALUATIONS_TABLE = "evaluations.json"
LABELS_TABLE = "labels.json"
DROPPED_TABLE = "dropped.json"
CONTROL_EVENTS_TABLE = "control-events.json"

#: MLflow 3.18 calls two pandas APIs that pandas 3 deprecates. Nothing here
#: can fix that, so the two messages are dropped around a table write instead
#: of printing once per eval in the training log.
TABLE_WARNINGS: tuple[str, ...] = (
    "For backward compatibility, 'str' dtypes are included by select_dtypes",
    "The 'convert_dates' keyword in read_json is deprecated",
)

#: The evaluation fields kept in the table. `prompt_ids` and `subset` are left
#: out, because a cell full of ids is unreadable.
EVALUATION_FIELDS: tuple[str, ...] = (
    "step",
    "kind",
    "stage",
    "count",
    "pass_at_1",
    "evaluation_seconds",
)

#: MLflow reads this setting when system metrics logging starts, so setting it
#: here covers the run the mirror opens next. A caller that set its own value
#: keeps it.
SYSTEM_METRICS_INTERVAL_ENV = "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"
SYSTEM_METRICS_INTERVAL = "1"

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

    def step(self, step: int) -> None: ...

    def eval(self, payload: Mapping[str, Any]) -> None: ...

    def labels(self) -> None: ...

    def gpu_sample(self, row: Mapping[str, Any]) -> None: ...

    def rebuild(self) -> None: ...

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


def _evaluation_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The evaluation points of `eval.json`, without the long id lists."""
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        return []
    rows: list[dict[str, Any]] = []
    for point in evaluations:
        if not isinstance(point, Mapping):
            continue
        rows.append({name: point[name] for name in EVALUATION_FIELDS if name in point})
    return rows


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

    def step(self, step: int) -> None:
        pass

    def eval(self, payload: Mapping[str, Any]) -> None:
        pass

    def labels(self) -> None:
        pass

    def gpu_sample(self, row: Mapping[str, Any]) -> None:
        pass

    def rebuild(self) -> None:
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
        self._mlflow: Any = None
        self._client: Any = None
        self._question_steps: set[int] = set()
        self._distribution_steps: set[int] = set()
        self._logged_rows: dict[str, int] = {
            EVALUATIONS_TABLE: 0,
            LABELS_TABLE: 0,
            DROPPED_TABLE: 0,
            CONTROL_EVENTS_TABLE: 0,
        }
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

        options: dict[str, Any] = {}
        if importlib.util.find_spec("psutil") is not None:
            options["log_system_metrics"] = True
        self._mlflow.set_experiment(EXPERIMENT_NAME)
        active = self._mlflow.start_run(run_name=name, **options)
        self.run_id = active.info.run_id

        self._mlflow.set_tags(tags)
        params = _params(meta)
        if params:
            self._mlflow.log_params(params)

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
        self._write_ids()

    @_guarded
    def step(self, step: int) -> None:
        """Copy the newest value of each series at `step`. Nulls are skipped."""
        client = self._require_client()
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        dashboard = collect_dashboard([self.run_dir])
        runs = dashboard.get("runs") or []
        if not runs:
            return
        series = runs[0].get("series") or {}
        index = int(step)
        for name, values in series.items():
            if name == "steps" or not isinstance(values, list) or not values:
                continue
            value = _number(values[-1])
            if value is None:
                continue
            client.log_metric(self.run_id, name, value, step=index)

    @_guarded
    def eval(self, payload: Mapping[str, Any]) -> None:
        """Log the run tables for the evaluation in `payload`."""
        self.push_tables(_payload_step(payload))

    def labels(self) -> None:
        """Placeholder. The label table is pushed with the other tables."""
        pass

    @_guarded
    def push_tables(self, step: int) -> None:
        """Log the run tables for one evaluation.

        The questions and distributions tables get one file per step, so a
        later eval does not replace an earlier one. The event tables append,
        and each call logs only the rows that arrived since the last call. A
        table with no rows is not written. One failed table is logged and the
        rest are still pushed.
        """
        self._require_client()
        if not self.run_id:
            raise RuntimeError("the parent run has not started")
        index = int(step)
        dashboard = collect_dashboard([self.run_dir])
        runs = dashboard.get("runs") or []
        run = runs[0] if runs else {}

        questions = (run.get("questions") or {}).get("rows") or []
        if index not in self._question_steps:
            if self._push(f"{QUESTIONS_DIR}/step-{index}.json", questions):
                self._question_steps.add(index)

        distributions = run.get("distributions") or {}
        if index not in self._distribution_steps:
            rows = _distribution_rows(distributions)
            if self._push(f"{DISTRIBUTIONS_DIR}/step-{index}.json", rows):
                self._distribution_steps.add(index)

        self._push_appended(
            EVALUATIONS_TABLE, _evaluation_rows(_read_json(self.run_dir / EVAL_NAME))
        )
        self._push_appended(LABELS_TABLE, _read_jsonl(self.run_dir / LABELS_NAME))
        self._push_appended(DROPPED_TABLE, _dropped_rows(run.get("dropped") or {}))
        self._push_appended(
            CONTROL_EVENTS_TABLE, _read_jsonl(self.run_dir / CONTROL_EVENTS_NAME)
        )

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

    def _push_appended(
        self, artifact_file: str, rows: list[Mapping[str, Any]]
    ) -> None:
        """Append the rows this run has not pushed yet."""
        seen = self._logged_rows.get(artifact_file, 0)
        if seen > len(rows):
            seen = 0
        fresh = rows[seen:]
        if not fresh:
            self._logged_rows[artifact_file] = len(rows)
            return
        if self._push(artifact_file, fresh):
            self._logged_rows[artifact_file] = len(rows)

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

    def rebuild(self) -> None:
        """Placeholder. The resume rebuild arrives with a later task."""
        pass

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
        payload = {"run_id": self.run_id, "gpu_run_id": self.gpu_run_id}
        _write_atomic(self.run_dir / IDS_NAME, json.dumps(payload, indent=2) + "\n")


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
