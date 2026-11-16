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


def test_dashboard_shows_the_prompt_and_sequence_counts_per_step(tmp_path) -> None:
    rows = [
        {"step": 1, "arm": "code-score", "prompts": 2, "sequences": 16, "mean_reward": 0.5},
        {"step": 2, "arm": "code-score", "prompts": 3, "sequences": 24, "mean_reward": 0.6},
    ]
    run_dir = write_run(tmp_path / "runs", "code-score", rows=rows)
    data = collect_dashboard([run_dir])

    run = data["runs"][0]
    assert run["latest"]["prompts"] == 3
    assert run["latest"]["sequences"] == 24
    assert run["series"]["prompts"] == [2, 3]
    assert run["series"]["sequences"] == [16, 24]

    html = render_html(data)
    assert "prompts per step" in html
    assert "sequences per step" in html
    assert "prompts and sequences per step" in html


def test_reward_histogram_covers_negative_rewards(tmp_path) -> None:
    rollouts = [
        {"step": 1, "reward": -1.0, "tests_passed": 0, "tests_total": 4},
        {"step": 1, "reward": -0.2, "tests_passed": 0, "tests_total": 4},
        {"step": 1, "reward": 1.0, "tests_passed": 4, "tests_total": 4},
    ]
    run_dir = write_run(tmp_path / "runs", "code-score", rollouts=rollouts)
    histogram = collect_dashboard([run_dir])["runs"][0]["distributions"]["reward"]

    assert histogram["bins"][0] == pytest.approx(-1.0)
    assert histogram["bins"][-1] == pytest.approx(0.9)
    assert len(histogram["counts"]) == 20
    assert histogram["counts"][0] == 1
    assert histogram["counts"][8] == 1
    assert histogram["counts"][19] == 1
    assert sum(histogram["counts"]) == 3


def test_stage_progress_and_band_rates_read_the_whole_rollout_file(tmp_path) -> None:
    """A prompt solved at the top of a long file still counts."""
    from rlvr.dashboard import ROLLOUT_TAIL_BYTES

    solved = [
        {
            "step": 1,
            "stage": "very easy",
            "prompt_id": "early-solved",
            "tests_passed": 4,
            "tests_total": 4,
            "reward": 1.0,
        }
        for _ in range(8)
    ]
    filler = [
        {
            "step": 2,
            "stage": "very easy",
            "prompt_id": "still-failing",
            "tests_passed": 0,
            "tests_total": 4,
            "reward": 0.0,
            "pad": "x" * 200,
        }
        for _ in range(20000)
    ]
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={"curriculum": {"stages": [{"name": "very easy", "train": 4}]}},
        rollouts=solved + filler,
    )
    assert (run_dir / "rollouts.jsonl").stat().st_size > ROLLOUT_TAIL_BYTES

    run = collect_dashboard([run_dir])["runs"][0]
    progress = run["stage_progress"]

    assert progress["solved"] == 1
    assert progress["total"] == 4
    assert progress["percent"] == pytest.approx(0.25)
    assert run["bands"] == [
        {"band": "very easy", "rollouts": 20008, "pass_rate": pytest.approx(8 / 20008)}
    ]


def test_a_stage_evaluation_is_surfaced_with_its_band(tmp_path) -> None:
    evaluations = [
        {
            "step": 1,
            "kind": "stage",
            "stage": "very easy",
            "pass_at_1": 0.5,
            "count": 2,
        },
        {"step": 2, "kind": "final", "pass_at_1": 0.6, "count": 2},
    ]
    run_dir = write_run(tmp_path / "runs", "code-score", evaluations=evaluations)
    data = collect_dashboard([run_dir])

    points = data["runs"][0]["evaluations"]
    assert points[0]["stage"] == "very easy"
    assert points[1]["stage"] is None
    assert "very easy" in render_html(data)


def test_dashboard_shows_the_active_stage_and_pass_rate_per_band(tmp_path) -> None:
    rows = [
        {
            "step": 1,
            "arm": "code-score",
            "stage": "very easy",
            "rollouts": 2,
            "mean_reward": 0.75,
            "pass_rate": 0.5,
            "mean_completion_tokens": 40.0,
            "cache_hits": 0,
            "truncation_rate": 0.0,
        },
        {
            "step": 2,
            "arm": "code-score",
            "stage": "medium",
            "rollouts": 1,
            "mean_reward": 1.0,
            "pass_rate": 1.0,
            "mean_completion_tokens": 30.0,
            "cache_hits": 0,
            "truncation_rate": 0.0,
        },
    ]
    rollouts = [
        {"step": 1, "stage": "very easy", "reward": 1.0, "tests_passed": 2, "tests_total": 2},
        {"step": 1, "stage": "very easy", "reward": 0.5, "tests_passed": 1, "tests_total": 2},
        {"step": 2, "stage": "medium", "reward": 1.0, "tests_passed": 2, "tests_total": 2},
    ]
    run_dir = write_run(tmp_path / "runs", "code-score", rows=rows, rollouts=rollouts)
    data = collect_dashboard([run_dir])
    run = data["runs"][0]

    assert run["stage"] == "medium"
    assert run["bands"] == [
        {"band": "very easy", "rollouts": 2, "pass_rate": 0.5},
        {"band": "medium", "rollouts": 1, "pass_rate": 1.0},
    ]
    html = render_html(data)
    assert "pass rate per band" in html
    assert "very easy" in html


def test_a_waiting_run_has_no_stage(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    (runs_root / "code-score").mkdir(parents=True)
    data = collect_dashboard([runs_root])
    assert data["runs"][0]["stage"] is None
    assert data["runs"][0]["bands"] == []


def test_collect_dashboard_ignores_a_truncated_final_metric_line(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score", truncated=True)
    data = collect_dashboard([run_dir])
    run = data["runs"][0]

    assert run["steps_done"] == 2
    assert run["latest"]["mean_reward"] == pytest.approx(0.75)
    assert run["series"]["steps"] == [1, 2]


def test_collect_dashboard_separates_the_memory_figures(tmp_path) -> None:
    rows = [
        {
            "step": 1,
            "arm": "code-score",
            "rollouts": 8,
            "mean_reward": 0.5,
            "pass_rate": 0.5,
            "mean_completion_tokens": 80.0,
            "cache_hits": 0,
            "truncation_rate": 0.0,
            "gpu_live_mib": 100.0,
            "gpu_cached_mib": 20.0,
            "gpu_fragmented_mib": 5.0,
            "gpu_total_mib": 1000.0,
            "gpu_unaccounted_mib": 875.0,
        }
    ]
    run_dir = write_run(tmp_path / "runs", "code-score", rows=rows)
    data = collect_dashboard([run_dir])
    run = data["runs"][0]

    assert run["latest"]["gpu_live_mib"] == pytest.approx(100.0)
    assert run["latest"]["gpu_cached_mib"] == pytest.approx(20.0)
    assert run["latest"]["gpu_fragmented_mib"] == pytest.approx(5.0)
    assert run["latest"]["gpu_total_mib"] == pytest.approx(1000.0)
    assert run["latest"]["gpu_unaccounted_mib"] == pytest.approx(875.0)
    assert run["series"]["gpu_live_mib"] == [pytest.approx(100.0)]

    html = render_html(data)
    for label in (
        "gpu live GiB",
        "gpu cached GiB",
        "gpu fragmented GiB",
        "gpu unaccounted GiB",
        "gpu total GiB",
    ):
        assert label in html


def test_a_fake_run_without_a_gpu_shows_zero_memory(run_fake) -> None:
    run_dir, _ = run_fake("code-score")
    data = collect_dashboard([run_dir])
    latest = data["runs"][0]["latest"]

    assert latest["gpu_live_mib"] == 0.0
    assert latest["gpu_cached_mib"] == 0.0
    assert latest["gpu_fragmented_mib"] == 0.0
    assert latest["gpu_total_mib"] == 0.0
    assert "gpu memory per step (GiB)" in render_html(data)


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
        meta_extra={
            "mode": "real",
            "config": {"training": {"epochs": 5}, "data": {"train_size": 714}},
        },
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 3570


def test_target_steps_divide_the_epoch_into_prompt_batches(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={
            "mode": "real",
            "trainer": {"epochs": 5, "batch_start_prompts": 2},
            "config": {"training": {"epochs": 5}, "data": {"train_size": 714}},
        },
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 1785


def test_target_steps_fall_back_to_the_real_formula_without_mode(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={"config": {"training": {"epochs": 5}, "data": {"train_size": 714}}},
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 3570


def test_target_steps_for_a_fake_run_uses_the_fake_step_count(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={
            "mode": "fake",
            "config": {
                "training": {"epochs": 5},
                "data": {"train_size": 714},
                "fake": {"steps": 4},
            },
        },
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 4


def test_target_steps_for_a_fake_run_is_capped_by_the_budget(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={
            "mode": "fake",
            "config": {"fake": {"steps": 4}, "budget": {"max_steps": 2}},
        },
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 2


def test_target_steps_reads_epochs_from_the_trainer_table(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={
            "mode": "real",
            "trainer": {"epochs": 3},
            "config": {"data": {"train_size": 714}},
        },
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 2142


def test_target_steps_capped_by_budget_max_steps(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={
            "mode": "real",
            "config": {
                "training": {"epochs": 5},
                "data": {"train_size": 714},
                "budget": {"max_steps": 100},
            },
        },
    )
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] == 100


def test_target_steps_absent_when_unknown(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["target_steps"] is None


def test_started_is_shown_as_a_readable_utc_timestamp(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score", timestamp="20261112T013045Z")
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["started"] == "20261112T013045Z"
    assert data["runs"][0]["started_utc"] == "2026-11-12 01:30:45Z"

    html = render_html(data)
    assert "2026-11-12 01:30:45Z" in html
    assert "started " in html


def test_an_unparsable_started_value_is_shown_as_is(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score", timestamp="not-a-time")
    data = collect_dashboard([run_dir])
    assert data["runs"][0]["started_utc"] == "not-a-time"


def test_gpu_read_parses_the_csv_output(monkeypatch) -> None:
    from rlvr import dashboard as dashboard_module

    def fake_run(command, **kwargs):
        return dashboard_module.subprocess.CompletedProcess(
            args=command, returncode=0, stdout="20, 19610, 49140, 41\n", stderr=""
        )

    monkeypatch.setattr(dashboard_module.subprocess, "run", fake_run)
    sample = dashboard_module._gpu_read()

    assert sample is not None
    assert sample["utilization"] == 20
    assert sample["memory_used"] == 19610
    assert sample["memory_total"] == 49140
    assert sample["temperature"] == 41


def test_gpu_read_returns_none_for_unsupported_fields(monkeypatch) -> None:
    from rlvr import dashboard as dashboard_module

    def fake_run(command, **kwargs):
        return dashboard_module.subprocess.CompletedProcess(
            args=command, returncode=0, stdout="[N/A], [N/A], 49140, 41\n", stderr=""
        )

    monkeypatch.setattr(dashboard_module.subprocess, "run", fake_run)
    sample = dashboard_module._gpu_read()

    assert sample is not None
    assert sample["utilization"] is None
    assert sample["memory_used"] is None
    assert sample["memory_total"] == 49140
    assert sample["temperature"] == 41


def test_gpu_sample_is_reused_when_younger_than_two_seconds(monkeypatch) -> None:
    from rlvr import dashboard as dashboard_module

    dashboard_module._gpu_history.clear()
    calls: list[int] = []

    def fake_read() -> dict:
        calls.append(1)
        return {
            "utilization": 10.0,
            "memory_used": 20.0,
            "memory_total": 30.0,
            "temperature": 40.0,
        }

    monkeypatch.setattr(dashboard_module, "_gpu_read", fake_read)
    try:
        dashboard_module.sample_gpu(now=100.0)
        dashboard_module.sample_gpu(now=101.9)
        assert len(calls) == 1

        dashboard_module.sample_gpu(now=102.0)
        assert len(calls) == 2
        assert len(dashboard_module._gpu_history) == 2
    finally:
        dashboard_module._gpu_history.clear()


def test_the_404_route_does_not_sample_the_gpu(monkeypatch) -> None:
    from rlvr import dashboard as dashboard_module

    handler = dashboard_module._DashboardHandler.__new__(
        dashboard_module._DashboardHandler
    )
    handler.path = "/missing"
    sent: list[tuple] = []
    monkeypatch.setattr(handler, "_send", lambda *args: sent.append(args))

    def refuse() -> None:
        raise AssertionError("the 404 route must not sample the GPU")

    monkeypatch.setattr(dashboard_module, "sample_gpu", refuse)
    handler._route()

    assert sent and sent[0][0] == 404


def test_the_data_route_samples_the_gpu(monkeypatch) -> None:
    from rlvr import dashboard as dashboard_module

    handler = dashboard_module._DashboardHandler.__new__(
        dashboard_module._DashboardHandler
    )
    handler.path = "/data.json"
    handler.paths = []
    sent: list[tuple] = []
    monkeypatch.setattr(handler, "_send", lambda *args: sent.append(args))
    calls: list[int] = []
    monkeypatch.setattr(dashboard_module, "sample_gpu", lambda: calls.append(1))

    handler._route()

    assert calls == [1]
    assert sent and sent[0][0] == 200


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


def test_render_html_is_self_contained_and_has_no_cdn() -> None:
    html = render_html({"generated_at": "now", "runs": [], "gpu": {"current": None, "history": []}})

    assert html.startswith("<!DOCTYPE html>")
    assert "cdn.tailwindcss.com" not in html
    assert "chart.umd" not in html
    assert "<script src=" not in html
    assert "<link" not in html
    assert '<div id="root"></div>' in html
    # The bundle is inlined, so the page draws without a second request.
    assert '<script type="module">' in html
    assert html.count('id="initial-data"') == 1


def test_the_page_carries_the_payload_as_parseable_json(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    html = render_html(collect_dashboard([run_dir]))

    start = html.index('id="initial-data">') + len('id="initial-data">')
    end = html.index("</script>", start)
    payload = json.loads(html[start:end])

    assert payload["runs"][0]["arm"] == "code-score"
    assert payload["runs"][0]["latest"]["mean_reward"] == pytest.approx(0.75)


def test_the_built_page_ships_in_the_package() -> None:
    from rlvr import dashboard as dashboard_module

    page = dashboard_module.WEB_DIR / dashboard_module.PAGE_NAME
    assert page.is_file()
    assert "__RLVR_INITIAL_DATA__" in page.read_text(encoding="utf-8")


def test_the_metrics_age_is_reported_in_seconds(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    run = collect_dashboard([run_dir])["runs"][0]

    assert run["metrics_age_seconds"] is not None
    assert run["metrics_age_seconds"] >= 0

    waiting_root = tmp_path / "waiting"
    (waiting_root / "code-score").mkdir(parents=True)
    waiting = collect_dashboard([waiting_root])["runs"][0]
    assert waiting["metrics_age_seconds"] is None


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
        "started_utc",
        "elapsed_seconds",
        "status",
        "steps_done",
        "target_steps",
        "latest",
        "series",
        "pass_at_1",
        "evaluations",
        "distributions",
        "questions",
    }
    assert expected <= set(run)
    assert decoded["generated_at"]
    assert "gpu" in decoded


def test_sparse_entropy_points_are_kept_as_gaps_not_zero(tmp_path) -> None:
    rows = [
        {
            "step": 1,
            "arm": "code-score",
            "rollouts": 8,
            "mean_reward": 0.5,
            "pass_rate": 0.5,
            "mean_entropy": 0.8,
            "mean_completion_tokens": 80.0,
            "cache_hits": 0,
            "truncation_rate": 0.0,
        },
        {
            "step": 2,
            "arm": "code-score",
            "rollouts": 8,
            "mean_reward": 0.6,
            "pass_rate": 0.6,
            "mean_completion_tokens": 70.0,
            "cache_hits": 0,
            "truncation_rate": 0.0,
        },
        {
            "step": 3,
            "arm": "code-score",
            "rollouts": 8,
            "mean_reward": 0.7,
            "pass_rate": 0.7,
            "mean_entropy": 0.7,
            "mean_completion_tokens": 60.0,
            "cache_hits": 0,
            "truncation_rate": 0.0,
        },
    ]
    run_dir = write_run(tmp_path / "runs", "code-score", rows=rows)
    data = collect_dashboard([run_dir])
    run = data["runs"][0]

    assert run["series"]["mean_entropy"] == [0.8, None, 0.7]
    # The KPI shows the last measurement, not a dash on a step without one.
    assert run["latest"]["mean_entropy"] == pytest.approx(0.7)

    html = render_html(data)
    # The entropy chart keeps the sampled series and drops the gaps.
    assert "entropy per step" in html


def test_latest_entropy_falls_back_to_the_last_measured_step(tmp_path) -> None:
    rows = [
        {
            "step": 1,
            "arm": "code-score",
            "rollouts": 8,
            "mean_reward": 0.5,
            "pass_rate": 0.5,
            "mean_entropy": 0.8,
            "mean_completion_tokens": 80.0,
        },
        {
            "step": 2,
            "arm": "code-score",
            "rollouts": 8,
            "mean_reward": 0.6,
            "pass_rate": 0.6,
            "mean_completion_tokens": 70.0,
        },
    ]
    run = collect_dashboard(
        [write_run(tmp_path / "runs", "code-score", rows=rows)]
    )["runs"][0]
    assert run["latest"]["mean_entropy"] == pytest.approx(0.8)


def test_latest_entropy_is_absent_when_no_step_measured(tmp_path) -> None:
    rows = [
        {
            "step": 1,
            "arm": "code-score",
            "rollouts": 8,
            "mean_reward": 0.5,
            "pass_rate": 0.5,
            "mean_completion_tokens": 80.0,
        }
    ]
    run = collect_dashboard(
        [write_run(tmp_path / "runs", "code-score", rows=rows)]
    )["runs"][0]
    assert run["latest"]["mean_entropy"] is None
    assert run["series"]["mean_entropy"] == [None]


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


def stage_rollouts(step: int, prompt_id: str, passes: int, total: int = 8) -> list[dict]:
    """One prompt group of 8 rollouts, `passes` of them fully passing."""
    return [
        {
            "step": step,
            "stage": "very easy",
            "prompt_id": prompt_id,
            "tests_passed": 3 if index < passes else 0,
            "tests_total": 3,
            "reward": 1.0 if index < passes else 0.0,
        }
        for index in range(total)
    ]


def test_stage_progress_counts_only_prompts_where_every_rollout_passed(tmp_path) -> None:
    rollouts = (
        stage_rollouts(1, "p1", passes=8)
        + stage_rollouts(1, "p2", passes=7)
        + stage_rollouts(2, "p2", passes=8)
    )
    run_dir = write_run(
        tmp_path / "runs",
        "code-score",
        meta_extra={
            "curriculum": {
                "stages": [
                    {"name": "very easy", "train": 4, "held_out": 1},
                    {"name": "medium", "train": 10, "held_out": 2},
                ]
            }
        },
        rollouts=rollouts,
    )
    data = collect_dashboard([run_dir])
    progress = data["runs"][0]["stage_progress"]

    assert progress["band"] == "very easy"
    assert progress["total"] == 4
    assert progress["solved"] == 2
    assert progress["percent"] == pytest.approx(0.5)
    assert [point["percent"] for point in progress["points"]][-1] == pytest.approx(0.5)


def test_stage_progress_is_blank_without_a_pool_size(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score", rollouts=stage_rollouts(1, "p1", 8))
    data = collect_dashboard([run_dir])
    progress = data["runs"][0]["stage_progress"]

    assert progress["points"] == []
    assert progress["percent"] is None
    html = render_html(data)
    assert "stage completion per step (%)" in html
    assert "stage done" in html


def question_group(step: int, prompt_id: str, passes: int, error: str = "runtime_error") -> list[dict]:
    """Eight rollouts of one prompt, `passes` of them fully passing three tests."""
    return [
        {
            "step": step,
            "stage": "very easy",
            "prompt_id": prompt_id,
            "tests_passed": 3 if index < passes else 0,
            "tests_total": 3,
            "error_class": None if index < passes else error,
            "reward": 1.0 if index < passes else 0.0,
        }
        for index in range(8)
    ]


def test_questions_aggregate_per_prompt_and_only_full_pass_steps_are_solved(tmp_path) -> None:
    rollouts = (
        question_group(1, "solved", passes=8)
        + question_group(1, "almost", passes=7)
        + question_group(2, "almost", passes=7)
        + question_group(1, "dead", passes=0)
        + question_group(2, "late", passes=8, error="timeout")
    )
    run_dir = write_run(tmp_path / "runs", "code-score", rollouts=rollouts)
    questions = collect_dashboard([run_dir])["runs"][0]["questions"]

    assert questions["total"] == 4
    rows = {row["prompt_id"]: row for row in questions["rows"]}

    assert rows["solved"]["solved"] is True
    assert rows["solved"]["steps"] == 1
    assert rows["solved"]["rollouts"] == 8
    assert rows["solved"]["full_pass_steps"] == 1
    assert rows["solved"]["pass_rate"] == pytest.approx(1.0)
    assert rows["solved"]["best_passed"] == 3
    assert rows["solved"]["best_total"] == 3
    assert rows["solved"]["last_step"] == 1
    assert rows["solved"]["last_error"] is None

    # Seven of eight passing is not solved. Two steps sampled, no full pass.
    assert rows["almost"]["solved"] is False
    assert rows["almost"]["steps"] == 2
    assert rows["almost"]["rollouts"] == 16
    assert rows["almost"]["full_pass_steps"] == 0
    assert rows["almost"]["pass_rate"] == pytest.approx(14 / 16)
    assert rows["almost"]["last_step"] == 2

    assert rows["dead"]["solved"] is False
    assert rows["dead"]["full_pass_steps"] == 0
    assert rows["dead"]["pass_rate"] == pytest.approx(0.0)
    assert rows["dead"]["last_error"] == "runtime_error"

    assert rows["late"]["solved"] is True
    assert rows["late"]["last_step"] == 2
    assert rows["late"]["last_error"] is None

    # Zero-pass rows come first, so dead questions are easy to spot.
    assert questions["rows"][0]["prompt_id"] in {"almost", "dead"}
    assert questions["rows"][0]["full_pass_steps"] == 0


def test_questions_report_prompt_text_when_a_question_file_is_present(tmp_path, monkeypatch) -> None:
    from rlvr import dashboard as dashboard_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "mbpp_full.jsonl").write_text(
        json.dumps({"task_id": 269, "text": "Write a function to find the ascii value."}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dashboard_module, "QUESTION_SOURCES", (("mbpp", "mbpp_full.jsonl"),))

    rollouts = [
        {
            "step": 1,
            "stage": "very easy",
            "prompt_id": "mbpp-269",
            "tests_passed": 3,
            "tests_total": 3,
            "reward": 1.0,
        }
    ] * 8
    run_dir = write_run(tmp_path / "runs", "code-score", rollouts=rollouts)
    row = collect_dashboard([run_dir])["runs"][0]["questions"]["rows"][0]

    assert row["text"] == "Write a function to find the ascii value."


def test_questions_do_not_fail_when_a_question_file_is_absent(tmp_path, monkeypatch) -> None:
    from rlvr import dashboard as dashboard_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dashboard_module, "QUESTION_SOURCES", (("mbpp", "missing.jsonl"),))
    run_dir = write_run(tmp_path / "runs", "code-score", rollouts=question_group(1, "mbpp-269", 8))
    row = collect_dashboard([run_dir])["runs"][0]["questions"]["rows"][0]

    assert row["prompt_id"] == "mbpp-269"
    assert row["text"] is None


def test_gpu_tiles_are_reported_in_gib_not_mib(tmp_path) -> None:
    rows = [
        {
            "step": 1,
            "arm": "code-score",
            "gpu_live_mib": 2048.0,
            "gpu_cached_mib": 1024.0,
            "gpu_fragmented_mib": 0.0,
            "gpu_unaccounted_mib": 4096.0,
            "gpu_total_mib": 49152.0,
        }
    ]
    run_dir = write_run(tmp_path / "runs", "code-score", rows=rows)
    html = render_html(collect_dashboard([run_dir]))

    assert "gpu live GiB" in html
    assert "gpu total GiB" in html
    assert "gpu memory per step (GiB)" in html
    assert "MiB" not in html
    assert "gpu unaccounted GiB" in html


def test_a_slow_step_is_still_running(tmp_path) -> None:
    """A step takes minutes, so a metrics file a few minutes old is live."""
    import os
    import time

    run_dir = write_run(tmp_path / "runs", "code-score")
    metrics = run_dir / "metrics.jsonl"
    old = time.time() - 300
    os.utime(metrics, (old, old))
    (run_dir / "eval.json").unlink()

    data = collect_dashboard([run_dir])
    assert data["runs"][0]["status"] == "running"


def test_an_abandoned_run_is_stopped(tmp_path) -> None:
    import os
    import time

    run_dir = write_run(tmp_path / "runs", "code-score")
    metrics = run_dir / "metrics.jsonl"
    old = time.time() - 7200
    os.utime(metrics, (old, old))
    (run_dir / "eval.json").unlink()

    data = collect_dashboard([run_dir])
    assert data["runs"][0]["status"] == "stopped"
