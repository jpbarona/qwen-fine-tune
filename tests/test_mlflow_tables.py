"""Tests for the run tables the mirror logs into MLflow."""

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
from rlvr.mlflow_log import MlflowMirror  # noqa: E402

# MLflow's SQLAlchemy store configures its mappers with a loader option that
# SQLAlchemy deprecates. The warning comes from the library, not this code.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)

RUN_NAME = "20260101T000000Z"


def make_run(
    tmp_path: Path,
    *,
    labels: list[dict] | None = None,
    control_events: list[dict] | None = None,
) -> Path:
    """A run folder with one row of every file the tables read."""
    run_dir = tmp_path / "runs" / "code-score" / RUN_NAME
    run_dir.mkdir(parents=True)
    meta = {
        "arm": "code-score",
        "timestamp": RUN_NAME,
        "pools": {"train": 2},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    metric = {"step": 1, "pass_rate": 1.0}
    (run_dir / "metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")

    rollout = {
        "step": 1,
        "prompt_id": "p1",
        "stage": "unstaged",
        "tests_passed": 2,
        "tests_total": 2,
        "reward": 1.0,
        "error_class": None,
    }
    (run_dir / "rollouts.jsonl").write_text(json.dumps(rollout) + "\n", encoding="utf-8")

    label_rows = (
        labels
        if labels is not None
        else [{"step": 1, "prompt_id": "p1", "label": "parked"}]
    )
    (run_dir / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows), encoding="utf-8"
    )

    if control_events is not None:
        (run_dir / "control-events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in control_events),
            encoding="utf-8",
        )

    evaluation = {
        "step": 1,
        "kind": "final",
        "stage": "unstaged",
        "count": 1,
        "pass_at_1": 1.0,
        "evaluation_seconds": 3.0,
        "prompt_ids": ["p1"],
        "subset": {"seed": 1},
    }
    payload = {
        "arm": "code-score",
        "timestamp": RUN_NAME,
        "step": 1,
        "pass_at_1": 1.0,
        "count": 1,
        "stop_reason": "completed",
        "evaluations": [evaluation],
    }
    (run_dir / "eval.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def set_eval_kind(run_dir: Path, kind: str) -> None:
    """Rewrite the newest evaluation's kind in `eval.json`."""
    path = run_dir / "eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluations"][-1]["kind"] = kind
    path.write_text(json.dumps(payload), encoding="utf-8")


def started_mirror(tmp_path: Path) -> tuple[MlflowMirror, Path]:
    """A started mirror over a fixture run folder."""
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    return mirror, run_dir


def read_rows(client: MlflowClient, run_id: str, artifact: str, dest: Path) -> list[dict]:
    """The rows of a logged table, downloaded from the store."""
    path = client.download_artifacts(run_id, artifact, dst_path=str(dest))
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(zip(payload["columns"], row)) for row in payload["data"]]


def test_each_eval_writes_a_questions_file_for_its_step(tmp_path: Path) -> None:
    mirror, _run_dir = started_mirror(tmp_path)
    mirror.push_tables(1)

    client = MlflowClient()
    paths = [item.path for item in client.list_artifacts(mirror.run_id, "questions")]
    assert any(path.endswith("step-1-final.json") for path in paths)

    rows = read_rows(
        client, mirror.run_id, "questions/step-1-final.json", tmp_path / "download"
    )
    assert [row["prompt_id"] for row in rows] == ["p1"]
    mirror.close()


def test_each_eval_writes_a_distributions_file_for_its_step(tmp_path: Path) -> None:
    mirror, _run_dir = started_mirror(tmp_path)
    mirror.push_tables(1)

    client = MlflowClient()
    paths = [item.path for item in client.list_artifacts(mirror.run_id, "distributions")]
    assert any(path.endswith("step-1-final.json") for path in paths)

    rows = read_rows(
        client, mirror.run_id, "distributions/step-1-final.json", tmp_path / "download"
    )
    assert {"reward_bin", "reward_count", "tests_passed_bin", "tests_passed_count"} <= set(
        rows[0]
    )
    mirror.close()


def test_a_second_eval_does_not_replace_the_first_questions_file(
    tmp_path: Path,
) -> None:
    mirror, _run_dir = started_mirror(tmp_path)
    mirror.push_tables(1)
    mirror.push_tables(2)

    client = MlflowClient()
    questions = sorted(
        item.path for item in client.list_artifacts(mirror.run_id, "questions")
    )
    assert questions == [
        "questions/step-1-final.json",
        "questions/step-2-final.json",
    ]
    distributions = sorted(
        item.path for item in client.list_artifacts(mirror.run_id, "distributions")
    )
    assert distributions == [
        "distributions/step-1-final.json",
        "distributions/step-2-final.json",
    ]
    mirror.close()


def test_two_evals_at_one_step_write_two_questions_files(tmp_path: Path) -> None:
    mirror, run_dir = started_mirror(tmp_path)
    set_eval_kind(run_dir, "checkpoint")
    mirror.push_tables(1)
    set_eval_kind(run_dir, "final")
    mirror.push_tables(1)

    client = MlflowClient()
    questions = sorted(
        item.path for item in client.list_artifacts(mirror.run_id, "questions")
    )
    assert questions == [
        "questions/step-1-checkpoint.json",
        "questions/step-1-final.json",
    ]
    distributions = sorted(
        item.path for item in client.list_artifacts(mirror.run_id, "distributions")
    )
    assert distributions == [
        "distributions/step-1-checkpoint.json",
        "distributions/step-1-final.json",
    ]
    mirror.close()


def test_step_logs_stage_and_band_metrics(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    rollout = {
        "step": 1,
        "prompt_id": "p2",
        "stage": "gated judge",
        "tests_passed": 1,
        "tests_total": 2,
        "reward": 0.5,
        "error_class": None,
    }
    with (run_dir / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rollout) + "\n")

    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    mirror.step(1)

    client = MlflowClient()
    solved = client.get_metric_history(mirror.run_id, "stage/unstaged/solved")
    assert [point.value for point in solved] == [1.0]
    total = client.get_metric_history(mirror.run_id, "stage/unstaged/total")
    assert [point.value for point in total] == [2.0]
    band = client.get_metric_history(mirror.run_id, "band/unstaged/pass_rate")
    assert [point.value for point in band] == [1.0]
    gated = client.get_metric_history(mirror.run_id, "band/gated_judge/pass_rate")
    assert [point.value for point in gated] == [0.0]
    mirror.close()


def test_a_fresh_mirror_does_not_re_append_the_event_tables(tmp_path: Path) -> None:
    mirror, run_dir = started_mirror(tmp_path)
    mirror.push_tables(1)
    mirror.close()

    counts = json.loads((run_dir / "mlflow.json").read_text(encoding="utf-8"))["tables"]
    assert counts["labels.json"] == 1
    assert counts["evaluations.json"] == 1

    fresh = MlflowMirror(tmp_path / "runs", run_dir)
    fresh.run_id = mirror.run_id
    fresh.push_tables(1)

    client = MlflowClient()
    labels = read_rows(client, mirror.run_id, "labels.json", tmp_path / "download")
    assert [row["prompt_id"] for row in labels] == ["p1"]
    evaluations = read_rows(
        client, mirror.run_id, "evaluations.json", tmp_path / "download"
    )
    assert len(evaluations) == 1
    fresh.close()
    assert not (run_dir / "mlflow-errors.log").exists()


def test_a_shrunk_event_file_does_not_re_push_the_whole_table(
    tmp_path: Path,
) -> None:
    rows = [
        {"step": 1, "prompt_id": "p1", "label": "parked"},
        {"step": 1, "prompt_id": "p2", "label": "train"},
    ]
    run_dir = make_run(tmp_path, labels=rows)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    mirror.push_tables(1)

    (run_dir / "labels.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    mirror.push_tables(2)

    client = MlflowClient()
    labels = read_rows(client, mirror.run_id, "labels.json", tmp_path / "download")
    assert [row["prompt_id"] for row in labels] == ["p1", "p2"]
    mirror.close()


def test_eval_json_is_logged_as_an_artifact(tmp_path: Path) -> None:
    mirror, _run_dir = started_mirror(tmp_path)
    mirror.push_tables(1)

    client = MlflowClient()
    names = {item.path for item in client.list_artifacts(mirror.run_id)}
    assert "eval.json" in names

    path = client.download_artifacts(
        mirror.run_id, "eval.json", dst_path=str(tmp_path / "download")
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["step"] == 1
    mirror.close()


def test_the_event_tables_append_without_repeating_rows(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    mirror.push_tables(1)

    labels_path = run_dir / "labels.jsonl"
    with labels_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 2, "prompt_id": "p2", "label": "train"}) + "\n")
    mirror.push_tables(2)
    mirror.push_tables(3)

    client = MlflowClient()
    labels = read_rows(client, mirror.run_id, "labels.json", tmp_path / "download")
    assert [row["prompt_id"] for row in labels] == ["p1", "p2"]
    evaluations = read_rows(
        client, mirror.run_id, "evaluations.json", tmp_path / "download"
    )
    assert len(evaluations) == 1
    assert evaluations[0]["prompt_ids"] == ["p1"]
    assert evaluations[0]["subset"] == {"seed": 1}
    mirror.close()


def test_an_empty_table_is_skipped(tmp_path: Path) -> None:
    mirror, _run_dir = started_mirror(tmp_path)
    mirror.push_tables(1)

    client = MlflowClient()
    names = {item.path for item in client.list_artifacts(mirror.run_id)}
    assert "dropped.json" not in names
    assert "control-events.json" not in names
    mirror.close()


def test_control_events_and_dropped_prompts_are_logged(tmp_path: Path) -> None:
    event = {"step": 1, "action": "apply", "drop": ["p2"]}
    run_dir = make_run(tmp_path, control_events=[event])
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    mirror.push_tables(1)

    client = MlflowClient()
    dropped = read_rows(client, mirror.run_id, "dropped.json", tmp_path / "download")
    assert dropped == [{"dropped_step": 1, "prompt_id": "p2"}]
    control = read_rows(
        client, mirror.run_id, "control-events.json", tmp_path / "download"
    )
    assert [row["action"] for row in control] == ["apply"]
    mirror.close()


def test_eval_pushes_the_tables_for_the_payload(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})
    payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    mirror.eval(payload)

    client = MlflowClient()
    paths = [item.path for item in client.list_artifacts(mirror.run_id, "questions")]
    assert any(path.endswith("step-1-final.json") for path in paths)
    mirror.close()


def test_a_failed_push_is_logged_and_not_raised(tmp_path: Path, monkeypatch) -> None:
    run_dir = make_run(tmp_path)
    mirror = MlflowMirror(tmp_path / "runs", run_dir)
    mirror.start({"arm": "code-score"})

    def explode(_paths):
        raise RuntimeError("tables exploded")

    monkeypatch.setattr(mlflow_log, "collect_dashboard", explode)
    mirror.push_tables(1)

    text = (run_dir / "mlflow-errors.log").read_text(encoding="utf-8")
    assert "tables exploded" in text
    mirror.close()


def test_one_failed_table_does_not_stop_the_others(
    tmp_path: Path, monkeypatch
) -> None:
    mirror, run_dir = started_mirror(tmp_path)
    real_log_table = mirror._mlflow.log_table

    def fail_questions(data, artifact_file, run_id):
        if artifact_file.startswith("questions/"):
            raise RuntimeError("questions table exploded")
        return real_log_table(data, artifact_file=artifact_file, run_id=run_id)

    monkeypatch.setattr(mirror._mlflow, "log_table", fail_questions)
    mirror.push_tables(1)

    text = (run_dir / "mlflow-errors.log").read_text(encoding="utf-8")
    assert "questions table exploded" in text
    client = MlflowClient()
    names = {item.path for item in client.list_artifacts(mirror.run_id)}
    assert "evaluations.json" in names
    assert "labels.json" in names
    mirror.close()
