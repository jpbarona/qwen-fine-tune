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

`eval`, `labels` and `rebuild` are placeholders until the tables, the resume
rebuild and the importer land.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .runread import collect_dashboard

__all__ = ["Mirror", "MlflowMirror", "NullMirror", "open_mirror"]

EXPERIMENT_NAME = "rlvr"
DATABASE_NAME = "mlflow.db"
ARTIFACTS_NAME = "mlartifacts"
IDS_NAME = "mlflow.json"
ERRORS_NAME = "mlflow-errors.log"

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

    def eval(self, payload: Mapping[str, Any]) -> None:
        """Placeholder. The tables arrive with a later task."""
        pass

    def labels(self) -> None:
        """Placeholder. The label table arrives with a later task."""
        pass

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
