"""Tests for the mirror that copies a run folder into a local MLflow store."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Silence a one time hint MLflow prints on import, so the test output stays
# readable.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.tracking import MlflowClient  # noqa: E402
import pytest  # noqa: E402

import rlvr.mlflow_log as mlflow_log  # noqa: E402
from rlvr.mlflow_log import MlflowMirror, NullMirror, open_mirror  # noqa: E402

# MLflow's SQLAlchemy store configures its mappers with a loader option that
# SQLAlchemy deprecates. The warning comes from the library, not this code.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)

RUN_NAME = "20260101T000000Z"


def make_run(tmp_path: Path, metrics: list[dict] | None = None) -> Path:
    """A run folder, with an optional metrics file, under a runs root."""
    run_dir = tmp_path / "runs" / "code-score" / RUN_NAME
    run_dir.mkdir(parents=True)
    if metrics is not None:
        text = "\n".join(json.dumps(row) for row in metrics) + "\n"
        (run_dir / "metrics.jsonl").write_text(text, encoding="utf-8")
    return run_dir


def test_start_writes_params_and_tags(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score", "git_commit": "abc", "model": {"name": "m"}})
    client = MlflowClient()
    run = client.get_run(mirror.run_id)
    assert run.data.tags["rlvr/arm"] == "code-score"
    assert run.data.tags["mlflow.runName"] == f"code-score-{RUN_NAME}"
    assert run.data.params["git_commit"] == "abc"
    assert run.data.params["model.name"] == "m"
    experiment = client.get_experiment(mirror.experiment_id)
    assert experiment.name == "rlvr"
    assert experiment.artifact_location == str(tmp_path / "runs" / "mlartifacts")
    mirror.close()
    assert client.get_run(mirror.run_id).info.status == "FINISHED"


def test_a_null_metric_is_skipped(tmp_path: Path) -> None:
    run_dir = make_run(
        tmp_path,
        [
            {"step": 1, "pass_rate": 0.25, "grad_norm": None},
            {"step": 2, "pass_rate": 0.5, "grad_norm": None},
        ],
    )
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    mirror.step({"step": 2, "pass_rate": 0.5, "grad_norm": None})

    client = MlflowClient()
    history = client.get_metric_history(mirror.run_id, "pass_rate")
    assert [(point.value, point.step) for point in history] == [(0.5, 2)]
    assert client.get_metric_history(mirror.run_id, "grad_norm") == []
    mirror.close()


def test_start_writes_the_run_ids_file(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})

    payload = json.loads((run_dir / "mlflow.json").read_text(encoding="utf-8"))
    assert payload == {
        "run_id": mirror.run_id,
        "gpu_run_id": mirror.gpu_run_id,
        "tables": {
            "evaluations.json": 0,
            "labels.json": 0,
            "dropped.json": 0,
            "control-events.json": 0,
        },
    }

    child = MlflowClient().get_run(mirror.gpu_run_id)
    assert child.data.tags["mlflow.parentRunId"] == mirror.run_id
    mirror.close()
    assert MlflowClient().get_run(mirror.gpu_run_id).info.status == "FINISHED"


def test_gpu_samples_land_on_the_child_run(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})

    mirror.gpu_sample({"t": 1.0, "utilization": 50.0, "temperature": None})
    mirror.gpu_sample(
        {
            "t": 2.0,
            "utilization": 0.0,
            "idle": 1,
            "count": 2,
            "age": 0.0,
        }
    )

    client = MlflowClient()
    history = client.get_metric_history(mirror.gpu_run_id, "utilization")
    assert [point.value for point in history] == [50.0, 0.0]
    assert [point.step for point in history] == [0, 1]
    assert client.get_metric_history(mirror.gpu_run_id, "t") == []
    assert client.get_metric_history(mirror.gpu_run_id, "temperature") == []
    for name, value in (("idle", 1.0), ("count", 2.0), ("age", 0.0)):
        points = client.get_metric_history(mirror.gpu_run_id, name)
        assert [point.value for point in points] == [value]
    mirror.close()


def test_a_failure_is_logged_and_not_raised(tmp_path: Path, monkeypatch) -> None:
    run_dir = make_run(tmp_path, [{"step": 1, "pass_rate": 0.5}])
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})

    def explode(self, _row, _step):
        raise RuntimeError("read exploded")

    monkeypatch.setattr(mlflow_log.MlflowMirror, "_log_row", explode)
    mirror.step({"step": 1, "pass_rate": 0.5})

    text = (run_dir / "mlflow-errors.log").read_text(encoding="utf-8")
    assert "read exploded" in text
    mirror.close()


def test_a_broken_store_is_logged_and_stays_quiet(tmp_path: Path, monkeypatch) -> None:
    run_dir = make_run(tmp_path, [{"step": 1, "pass_rate": 0.5}])

    def explode(_self):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(mlflow_log.MlflowMirror, "_connect", explode)
    mirror = open_mirror(tmp_path / "runs", run_dir, True)
    assert isinstance(mirror, NullMirror)
    mirror.start({"arm": "code-score"})
    mirror.step({"step": 1})
    mirror.gpu_sample({"utilization": 50.0})
    mirror.close()

    text = (run_dir / "mlflow-errors.log").read_text(encoding="utf-8")
    assert "store unavailable" in text
    assert text.count("store unavailable") == 1


def test_a_creation_race_re_reads_the_experiment_by_name() -> None:
    class RacingClient:
        def __init__(self) -> None:
            self.lookups = 0

        def get_experiment_by_name(self, name):
            self.lookups += 1
            if self.lookups == 1:
                return None
            return type("Experiment", (), {"experiment_id": "42"})()

        def create_experiment(self, name, artifact_location=None):
            raise RuntimeError("RESOURCE_ALREADY_EXISTS: experiment already exists")

    client = RacingClient()
    experiment_id = mlflow_log._ensure_experiment_id(client, "s3://bucket/artifacts")

    assert experiment_id == "42"
    assert client.lookups == 2


def test_a_creation_failure_that_is_not_a_race_still_raises() -> None:
    class BrokenClient:
        def get_experiment_by_name(self, name):
            return None

        def create_experiment(self, name, artifact_location=None):
            raise RuntimeError("permission denied")

    with pytest.raises(RuntimeError, match="permission denied"):
        mlflow_log._ensure_experiment_id(BrokenClient(), "s3://bucket/artifacts")


def test_open_mirror_respects_the_flag(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    assert isinstance(open_mirror(tmp_path / "runs", run_dir, False), NullMirror)
    mirror = open_mirror(tmp_path / "runs", run_dir, True)
    assert isinstance(mirror, MlflowMirror)
    mirror.close()


def test_the_null_mirror_accepts_every_call(tmp_path: Path) -> None:
    mirror = NullMirror()
    mirror.start({"arm": "code-score"})
    mirror.step({"step": 1})
    mirror.eval({"evaluations": []})
    mirror.labels()
    mirror.gpu_sample({"utilization": 50.0})
    mirror.rebuild()
    mirror.close()


def test_start_writes_the_step_target_as_max_steps(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start(
        {
            "arm": "code-score",
            "mode": "fake",
            "config": {"fake": {"steps": 5}},
        }
    )
    run = MlflowClient().get_run(mirror.run_id)
    assert run.data.params["max_steps"] == "5"
    mirror.close()


def test_mirror_uses_the_configured_tracking_uri(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    run_dir = make_run(tmp_path)
    uri = f"sqlite:///{store / 'other.db'}"
    mirror = MlflowMirror(tmp_path / "runs", run_dir, tracking_uri=uri)
    mirror.start({"arm": "code-score"})
    assert (store / "other.db").exists()
    assert not (tmp_path / "runs" / "mlflow.db").exists()
    mirror.close()


def test_mirror_creates_the_experiment_at_the_artifact_root(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    root = "s3://opencodeagent1/mlflow/v1/artifacts"
    mirror = MlflowMirror(tmp_path / "runs", run_dir, artifact_root=root)
    mirror.start({"arm": "code-score"})
    client = MlflowClient()
    experiment = client.get_experiment_by_name("rlvr")
    assert experiment.artifact_location == root
    mirror.close()


def test_mirror_without_a_uri_keeps_the_local_store(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    assert (tmp_path / "runs" / "mlflow.db").exists()
    mirror.close()


def test_eval_logs_both_pass_at_1_series(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    payload = {
        "step": 2,
        "evaluations": [
            {"step": 1, "kind": "checkpoint", "pass_at_1": 0.25},
            {"step": 2, "kind": "final", "pass_at_1": 0.5},
        ],
    }
    mirror.eval(payload)

    client = MlflowClient()
    checkpoint = client.get_metric_history(mirror.run_id, "pass_at_1/checkpoint")
    final = client.get_metric_history(mirror.run_id, "pass_at_1/final")
    assert [(point.step, point.value) for point in checkpoint] == [(1, 0.25)]
    assert [(point.step, point.value) for point in final] == [(2, 0.5)]
    mirror.close()
