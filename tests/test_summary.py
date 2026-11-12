"""Tests for summary aggregation over synthetic run folders.

No model, judge, Docker, or network is involved. Each run folder is a few JSONL
rows and one eval.json written by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.summary import (
    RunReport,
    StepMetric,
    SummaryError,
    format_summary,
    load_report,
    load_reports,
    render_html,
    resolve_run_dirs,
    write_report,
)

TIMESTAMP_A = "20261112T000000Z"
TIMESTAMP_B = "20261113T000000Z"


def write_run(
    root: Path,
    arm: str,
    *,
    timestamp: str = TIMESTAMP_A,
    rows: list[dict] | None = None,
    evaluations: list[dict] | None = None,
    final: dict | None = None,
    subset: dict | None = None,
    eval_file: bool = True,
) -> Path:
    run_dir = root / arm / timestamp
    run_dir.mkdir(parents=True)

    if rows is None:
        rows = [
            {
                "step": 1,
                "arm": arm,
                "rollouts": 2,
                "mean_reward": 0.5,
                "pass_rate": 0.5,
                "mean_entropy": 0.8,
                "mean_completion_tokens": 80.0,
                "cache_hits": 0,
            },
            {
                "step": 2,
                "arm": arm,
                "rollouts": 2,
                "mean_reward": 0.75,
                "pass_rate": 1.0,
                "mean_entropy": 0.6,
                "mean_completion_tokens": 60.0,
                "cache_hits": 0,
            },
        ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"arm": arm, "timestamp": timestamp}), encoding="utf-8"
    )

    if eval_file:
        if evaluations is None:
            evaluations = [
                {
                    "step": 1,
                    "kind": "checkpoint",
                    "pass_at_1": 0.25,
                    "count": 2,
                    "prompt_ids": ["a", "b"],
                    "subset": subset
                    or {
                        "method": "seeded_sample",
                        "seed": 9,
                        "requested": 100,
                        "selected": 2,
                        "pool_size": 2,
                    },
                }
            ]
        if final is None:
            final = {"step": 2, "pass_at_1": 0.5, "count": 2}
        payload = {
            "arm": arm,
            "timestamp": timestamp,
            "step": final["step"],
            "pass_at_1": final["pass_at_1"],
            "count": final["count"],
            "subset": subset,
            "evaluations": evaluations + [dict(final, kind="final")],
        }
        (run_dir / "eval.json").write_text(json.dumps(payload), encoding="utf-8")

    return run_dir


def test_load_report_reads_the_arm_metrics_and_final_eval(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    report = load_report(run_dir)

    assert report.arm == "code-score"
    assert report.timestamp == TIMESTAMP_A
    assert [step.step for step in report.steps] == [1, 2]
    assert report.steps[0].mean_reward == pytest.approx(0.5)
    assert report.steps[0].mean_entropy == pytest.approx(0.8)
    assert report.steps[0].mean_completion_tokens == pytest.approx(80.0)
    assert report.steps[0].pass_at_1 == pytest.approx(0.25)
    assert report.steps[1].pass_at_1 == pytest.approx(0.5)
    assert report.final_pass_at_1 == pytest.approx(0.5)
    assert report.final_count == 2


def test_load_report_uses_the_last_evaluation_at_a_step(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "hybrid",
        evaluations=[
            {"step": 2, "kind": "checkpoint", "pass_at_1": 0.1, "count": 2}
        ],
        final={"step": 2, "pass_at_1": 0.9, "count": 250},
    )
    report = load_report(run_dir)
    assert report.steps[-1].pass_at_1 == pytest.approx(0.9)


def test_a_run_without_an_eval_file_still_loads(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "hybrid", eval_file=False)
    report = load_report(run_dir)
    assert report.final_pass_at_1 is None
    assert report.final_count is None
    assert all(step.pass_at_1 is None for step in report.steps)


def test_a_run_without_metrics_is_an_error(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "hybrid" / TIMESTAMP_A
    run_dir.mkdir(parents=True)
    with pytest.raises(SummaryError, match="metrics.jsonl"):
        load_report(run_dir)


def test_a_malformed_metric_row_names_the_field(tmp_path) -> None:
    run_dir = write_run(
        tmp_path / "runs",
        "hybrid",
        rows=[{"step": 1, "mean_reward": "high"}],
    )
    with pytest.raises(SummaryError, match="mean_reward"):
        load_report(run_dir)


def test_resolve_accepts_a_run_folder(tmp_path) -> None:
    run_dir = write_run(tmp_path / "runs", "code-score")
    assert resolve_run_dirs([run_dir]) == [run_dir]


def test_resolve_expands_an_arm_folder_and_a_runs_root(tmp_path) -> None:
    root = tmp_path / "runs"
    first = write_run(root, "code-score")
    second = write_run(root, "hybrid")
    assert set(resolve_run_dirs([root])) == {first, second}
    assert resolve_run_dirs([root / "code-score"]) == [first]


def test_resolve_deduplicates_repeated_paths(tmp_path) -> None:
    root = tmp_path / "runs"
    run_dir = write_run(root, "code-score")
    assert resolve_run_dirs([run_dir, run_dir, root]) == [run_dir]


def test_resolve_rejects_a_folder_that_is_not_a_run(tmp_path) -> None:
    with pytest.raises(SummaryError, match="not a run folder"):
        resolve_run_dirs([tmp_path / "nothing"])


def test_load_reports_sorts_by_arm_order_then_timestamp(tmp_path) -> None:
    root = tmp_path / "runs"
    hybrid = write_run(root, "hybrid")
    code = write_run(root, "code-score")
    older = write_run(root, "code-score", timestamp="20260101T000000Z")
    reports = load_reports([root])
    assert [(report.arm, report.timestamp) for report in reports] == [
        ("code-score", "20260101T000000Z"),
        ("code-score", TIMESTAMP_A),
        ("hybrid", TIMESTAMP_A),
    ]
    assert older in {report.run_dir for report in reports}
    assert code in {report.run_dir for report in reports}
    assert hybrid in {report.run_dir for report in reports}


def test_format_summary_groups_arms_side_by_side() -> None:
    code = RunReport(
        arm="code-score",
        run_dir=Path("/runs/code-score/1"),
        timestamp="1",
        steps=(
            StepMetric(1, 0.5, 0.5, 0.8, 80.0, 2, None),
            StepMetric(2, 0.75, 1.0, 0.6, 60.0, 2, 0.5),
        ),
        final_pass_at_1=0.5,
        final_count=250,
        checkpoint_subset=None,
    )
    hybrid = RunReport(
        arm="hybrid",
        run_dir=Path("/runs/hybrid/1"),
        timestamp="1",
        steps=(StepMetric(1, 0.25, 0.25, 0.7, 70.0, 2, 0.25),),
        final_pass_at_1=0.25,
        final_count=250,
        checkpoint_subset=None,
    )

    text = format_summary((code, hybrid))

    for title in ("reward per step", "pass@1 per step", "entropy per step", "length per step"):
        assert title in text
    assert "code-score" in text and "hybrid" in text
    assert "0.5000" in text
    assert "0.8000" in text
    assert "80.0" in text
    assert "final pass@1 0.5000 over 250 questions" in text


def test_format_summary_uses_the_timestamp_when_an_arm_repeats() -> None:
    def report(timestamp: str) -> RunReport:
        return RunReport(
            arm="hybrid",
            run_dir=Path(f"/runs/hybrid/{timestamp}"),
            timestamp=timestamp,
            steps=(StepMetric(1, 0.5, 0.5, 0.8, 80.0, 2, None),),
            final_pass_at_1=None,
            final_count=None,
            checkpoint_subset=None,
        )

    text = format_summary((report("1"), report("2")))
    assert "hybrid@1" in text
    assert "hybrid@2" in text


def test_render_html_has_one_section_per_arm(tmp_path) -> None:
    root = tmp_path / "runs"
    write_run(root, "code-score")
    write_run(root, "gated-process")
    reports = load_reports([root])
    html = render_html(reports, generated_at="2026-11-12T00:00:00Z")

    assert html.startswith("<!DOCTYPE html>")
    assert html.count('<section class="arm">') == 2
    assert '<h2>code-score</h2>' in html
    assert '<h2>gated-process</h2>' in html
    assert "2026-11-12T00:00:00Z" in html
    assert "Final pass@1: 0.5000 over 2 questions" in html
    assert "seeded sample" in html
    assert "seed 9" in html


def test_render_html_is_static_and_self_contained(tmp_path) -> None:
    root = tmp_path / "runs"
    write_run(root, "code-score")
    html = render_html(load_reports([root]), generated_at="now")
    lowered = html.lower()
    assert "<script" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "animation" not in lowered
    assert "gradient" not in lowered


def test_render_html_escapes_text_from_a_run() -> None:
    report = RunReport(
        arm="<script>alert(1)</script>",
        run_dir=Path("/runs/<b>bad</b>"),
        timestamp="<i>1</i>",
        steps=(),
        final_pass_at_1=None,
        final_count=None,
        checkpoint_subset=None,
    )
    html = render_html((report,), generated_at="now")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>bad</b>" not in html


def test_write_report_writes_the_html_file(tmp_path) -> None:
    root = tmp_path / "runs"
    write_run(root, "code-score")
    out = tmp_path / "report.html"
    written = write_report(out, load_reports([root]), generated_at="now")
    assert written == out
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
