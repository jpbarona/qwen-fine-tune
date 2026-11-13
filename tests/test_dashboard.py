"""Tests for the live dashboard data collection and page shell.

No model, judge, Docker, or network is involved. Each run folder is a few JSONL
rows, one eval.json, and one meta.json written by hand. GPU sampling is never
run in these tests, so the GPU section stays empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.dashboard import collect_dashboard, render_html

TIMESTAMP = "20261112T000000Z"


def write_run(
    root: Path,
    arm: str = "code-score",
    *,
    timestamp: str = TIMESTAMP,
    rows: list[dict] | None = None,
    truncated: bool = False,
    evaluations: list[dict] | None = None,
    meta_extra: dict | None = None,
    rollouts: list[dict] | None = None,
) -> Path:
    run_dir = root / arm / timestamp
    run_dir.mkdir(parents=True)

    if rows is None:
        rows = [
            {
                "step": 1,
                "arm": arm,
                "rollouts": 8,
                "mean_reward": 0.5,
                "pass_rate": 0.5,
                "mean_entropy": 0.8,
                "mean_completion_tokens": 80.0,
                "cache_hits": 1,
                "truncation_rate": 0.1,
            },
            {
                "step": 2,
                "arm": arm,
                "rollouts": 8,
                "mean_reward": 0.75,
                "pass_rate": 1.0,
                "mean_entropy": 0.6,
                "mean_completion_tokens": 60.0,
                "cache_hits": 2,
                "truncation_rate": 0.2,
            },
        ]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if truncated:
        text += '{"step": 3, "mean_reward": 0.9, "pass_r'
    (run_dir / "metrics.jsonl").write_text(text, encoding="utf-8")

    meta = {"arm": arm, "timestamp": timestamp}
    if meta_extra:
        meta.update(meta_extra)
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    if evaluations is None:
        evaluations = [
            {"step": 1, "kind": "checkpoint", "pass_at_1": 0.25, "count": 2},
            {"step": 2, "kind": "final", "pass_at_1": 0.5, "count": 2},
        ]
    (run_dir / "eval.json").write_text(
        json.dumps({"arm": arm, "timestamp": timestamp, "evaluations": evaluations}),
        encoding="utf-8",
    )

    if rollouts is None:
        rollouts = [
            {"step": 1, "reward": 0.1, "tests_passed": 3, "tests_total": 5},
            {"step": 1, "reward": 0.9, "tests_passed": 5, "tests_total": 5},
            {"step": 2, "reward": 0.5, "tests_passed": 2, "tests_total": 5},
        ]
    (run_dir / "rollouts.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in rollouts), encoding="utf-8"
    )

    return run_dir


def test_collect_dashboard_reads_run_files(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    data = collect_dashboard([run_dir])

    assert data["runs"][0]["arm"] == "code-score"
    assert data["runs"][0]["run_name"] == TIMESTAMP
    assert data["runs"][0]["status"] == "finished"
    assert data["runs"][0]["steps_done"] == 2
    assert data["runs"][0]["latest"]["mean_reward"] == pytest.approx(0.75)
    assert data["runs"][0]["latest"]["cache_hits"] == 2
    kinds = [e["kind"] for e in data["runs"][0]["evaluations"]]
    assert kinds == ["checkpoint", "final"]
    assert data["runs"][0]["evaluations"][1]["pass_at_1"] == pytest.approx(0.5)


def test_collect_dashboard_ignores_a_truncated_final_metric_line(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score", truncated=True)
    data = collect_dashboard([run_dir])
    run = data["runs"][0]

    assert run["steps_done"] == 2
    assert run["latest"]["mean_reward"] == pytest.approx(0.75)
    assert run["series"]["steps"] == [1, 2]


def test_collect_dashboard_ignores_a_truncated_rollout_line(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    path = run_dir / "rollouts.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"step": 3, "reward": 0.2, "tests_p', encoding="utf-8")

    data = collect_dashboard([run_dir])
    counts = data["runs"][0]["distributions"]["reward"]["counts"]
    assert sum(counts) == 3


def test_collect_dashboard_missing_folder_is_empty(tmp_path) -> None:
    data = collect_dashboard([tmp_path / "nothing"])
    assert data["runs"] == []


def test_collect_dashboard_empty_arm_folder_shows_waiting(tmp_path) -> None:
    root = tmp_path / "runs"
    (root / "code-score").mkdir(parents=True)
    data = collect_dashboard([root])
    assert len(data["runs"]) == 1
    assert data["runs"][0]["arm"] == "code-score"
    assert data["runs"][0]["status"] == "waiting"


def test_collect_dashboard_runs_root_expands_to_latest_run(tmp_path) -> None:
    root = tmp_path / "runs"
    write_run(root, "code-score", timestamp="20260101T000000Z")
    newer = write_run(root, "code-score", timestamp=TIMESTAMP)
    data = collect_dashboard([root])
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_dir"] == str(newer)


def test_target_steps_derived_for_a_real_run(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={"config": {"training": {"epochs": 5}, "data": {"train_size": 714}}},
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 3570


def test_target_steps_capped_by_budget_max_steps(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={
            "config": {
                "training": {"epochs": 5},
                "data": {"train_size": 714},
                "budget": {"max_steps": 100},
            }
        },
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 100


def test_target_steps_absent_when_unknown(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] is None


def test_render_html_escapes_an_arm_name_with_a_script_tag() -> None:
    data = {
        "generated_at": "2026-11-13T00:00:00Z",
        "runs": [
            {
                "arm": "<script>alert(1)</script>",
                "run_name": "t",
                "run_dir": "runs/x/t",
                "started": "t",
                "elapsed_seconds": 1.0,
                "status": "running",
                "steps_done": 1,
                "target_steps": None,
                "latest": {},
                "series": {"steps": [], "mean_reward": []},
                "pass_at_1": {"checkpoint": {"steps": [], "values": []}, "final": {"steps": [], "values": []}},
                "evaluations": [],
                "distributions": {"reward": {"bins": [], "counts": []}, "tests_passed": {"bins": [], "counts": []}},
            }
        ],
        "gpu": {"current": None, "history": []},
    }
    html = render_html(data)

    assert "<script>alert(1)</script>" not in html
    assert "\\u003cscript\\u003e" in html


def test_render_html_is_self_contained_and_cdn_backed() -> None:
    html = render_html({"generated_at": "now", "runs": [], "gpu": {"current": None, "history": []}})

    assert html.startswith("<!DOCTYPE html>")
    assert "cdn.tailwindcss.com" in html
    assert "chart.js" in html
    assert "tailwind.config" in html


def test_dashboard_data_is_json_serializable(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    data = collect_dashboard([run_dir])
    encoded = json.dumps(data)
    decoded = json.loads(encoded)

    run = decoded["runs"][0]
    expected = {
        "arm",
        "run_name",
        "run_dir",
        "started",
        "elapsed_seconds",
        "status",
        "steps_done",
        "target_steps",
        "latest",
        "series",
        "pass_at_1",
        "evaluations",
        "distributions",
    }
    assert expected <= set(run)
    assert decoded["generated_at"]
    assert "gpu" in decoded


def test_series_and_distribution_shapes(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    run = collect_dashboard([run_dir])["runs"][0]

    assert run["series"]["steps"] == [1, 2]
    for name in ("mean_reward", "pass_rate", "mean_entropy", "mean_completion_tokens", "truncation_rate"):
        assert len(run["series"][name]) == 2

    reward = run["distributions"]["reward"]
    assert len(reward["bins"]) == 20
    assert len(reward["counts"]) == 20
    assert sum(reward["counts"]) == 3

    tests = run["distributions"]["tests_passed"]
    assert tests["bins"] == [0, 1, 2, 3, 4, 5]
    assert tests["counts"] == [0, 0, 1, 1, 0, 1]

    assert run["pass_at_1"]["checkpoint"]["steps"] == [1]
    assert run["pass_at_1"]["final"]["steps"] == [2]
