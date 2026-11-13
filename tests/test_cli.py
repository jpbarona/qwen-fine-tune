"""Tests for the command line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rlvr.cli import build_judge, build_prompts, main
from rlvr.config import load_config
from rlvr.data import load_split, prompt_id_for
from rlvr.fake import FakeJudge

CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 2
prompts_per_step = 2
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_run_fake_exits_zero_for_the_code_score_arm(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    assert main(["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]) == 0
    assert (runs_root / "code-score").is_dir()
    assert "runs/code-score" in capsys.readouterr().out


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_judge_arms_run_fake_and_populate_judge_scores(arm, config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    assert main(
        ["run", "--arm", arm, "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    run_dir = next((runs_root / arm).iterdir())
    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["process_score"] is None for record in records)
    for record in records:
        if record["judge_score"] is None:
            assert record["reward"] == 0.0
        else:
            assert 0.0 <= record["judge_score"] <= 1.0
            assert 0.0 <= record["reward"] <= 1.0
    assert any(record["judge_score"] is not None for record in records)


def test_gated_process_runs_fake_and_populates_both_scores(
    config_path, tmp_path
) -> None:
    runs_root = tmp_path / "runs"
    assert main(
        [
            "run",
            "--arm",
            "gated-process",
            "--fake",
            "--config",
            str(config_path),
            "--runs-root",
            str(runs_root),
        ]
    ) == 0
    run_dir = next((runs_root / "gated-process").iterdir())
    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    passing = [record for record in records if record["process_score"] is not None]
    assert passing
    for record in passing:
        assert 0.0 <= record["process_score"] <= 1.0
        assert 0.0 <= record["judge_score"] <= 1.0
        assert record["reward"] == pytest.approx(
            0.5 * record["process_score"] + 0.5 * record["judge_score"]
        )
    for record in records:
        if record["process_score"] is None:
            assert record["judge_score"] is None
            assert record["reward"] == 0.0


@pytest.mark.parametrize("arm", ["hybrid", "gated-judge"])
def test_a_judge_arm_fails_before_the_run_folder_when_the_key_is_missing(
    arm, config_path, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    runs_root = tmp_path / "runs"
    code = main(
        ["run", "--arm", arm, "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "OPENCODE_GO_API_KEY" in captured.err
    assert not (runs_root / arm).exists()


def test_build_judge_only_builds_a_client_for_arms_that_judge(
    write_config, monkeypatch
) -> None:
    from rlvr import cli

    built = []

    class StubClient:
        @classmethod
        def from_config(cls, config, **kwargs):
            built.append(config)
            return FakeJudge()

    monkeypatch.setattr(cli, "JudgeClient", StubClient)
    config = load_config(write_config())

    build_judge("code-score", config, fake=False)
    assert built == []

    build_judge("hybrid", config, fake=False)
    build_judge("gated-judge", config, fake=False)
    build_judge("gated-process", config, fake=False)
    assert built == [config, config, config]

    build_judge("hybrid", config, fake=True)
    assert built == [config, config, config]


def test_run_writes_the_four_files(config_path, tmp_path) -> None:
    runs_root = tmp_path / "runs"
    main(["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)])
    run_dir = next((runs_root / "code-score").iterdir())
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "checkpoints",
        "eval.json",
        "meta.json",
        "metrics.jsonl",
        "rollouts.jsonl",
    ]
    assert (run_dir / "meta.json").read_text(encoding="utf-8").startswith("{")


def test_unknown_arm_is_reported_on_stderr(config_path, tmp_path, capsys) -> None:
    code = main(
        [
            "run",
            "--arm",
            "reward-shaping",
            "--fake",
            "--config",
            str(config_path),
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "reward-shaping" in captured.err
    assert "code-score" in captured.err and "gated-process" in captured.err


def test_missing_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    code = main(
        ["run", "--arm", "code-score", "--fake", "--config", str(tmp_path / "absent.toml")]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "absent.toml" in captured.err
    assert "config" in captured.err


def test_invalid_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[judge]\nmodel = 5\n", encoding="utf-8")
    code = main(["run", "--arm", "code-score", "--fake", "--config", str(bad)])
    captured = capsys.readouterr()
    assert code != 0
    assert "config" in captured.err


def test_unparseable_config_is_reported_on_stderr(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not toml at all ===\n", encoding="utf-8")
    code = main(["run", "--arm", "code-score", "--fake", "--config", str(bad)])
    captured = capsys.readouterr()
    assert code != 0
    assert "config" in captured.err


def test_without_fake_the_command_asks_for_the_train_extra(
    config_path, tmp_path, capsys, monkeypatch
) -> None:
    import rlvr.qwen

    def missing(*args, **kwargs):
        raise ImportError("simulated missing torch")

    monkeypatch.setattr(rlvr.qwen, "QwenGenerator", missing)
    runs_root = tmp_path / "runs"
    code = main(
        ["run", "--arm", "code-score", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "train extra" in captured.err
    assert "--fake" in captured.err
    # The generator fails before the run folder is made, so nothing is left behind.
    assert not (runs_root / "code-score").exists()


def test_arm_is_required(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--fake", "--runs-root", str(tmp_path / "runs")])
    assert excinfo.value.code != 0
    assert "--arm" in capsys.readouterr().err


def test_unknown_subcommand_fails(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["train", "--arm", "code-score", "--fake"])


def test_runs_root_defaults_to_runs_in_the_current_directory(config_path, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--arm", "code-score", "--fake", "--config", str(config_path)]) == 0
    assert (tmp_path / "runs" / "code-score").is_dir()


def test_default_config_is_used_when_no_flag_is_given(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--arm", "code-score", "--fake", "--runs-root", str(tmp_path / "runs")]) == 0
    run_dir = next((tmp_path / "runs" / "code-score").iterdir())
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_revision"]


def test_console_script_runs_end_to_end(config_path, tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlvr",
            "run",
            "--arm",
            "code-score",
            "--fake",
            "--config",
            str(config_path),
            "--runs-root",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "runs" / "code-score").is_dir()


def test_console_script_reports_unknown_arm_without_a_traceback(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rlvr", "run", "--arm", "bogus", "--fake"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bogus" in result.stderr
    assert "Traceback" not in result.stderr


def test_summary_prints_the_metrics_and_writes_the_report(
    config_path, tmp_path, capsys
) -> None:
    runs_root = tmp_path / "runs"
    main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    capsys.readouterr()

    report = tmp_path / "report.html"
    code = main(["summary", str(runs_root), "--html", str(report)])

    printed = capsys.readouterr().out
    assert code == 0
    for title in ("reward per step", "pass@1 per step", "entropy per step", "length per step"):
        assert title in printed
    assert "code-score" in printed
    assert f"report: {report}" in printed
    assert report.is_file()
    assert "<!DOCTYPE html>" in report.read_text(encoding="utf-8")


def test_summary_accepts_more_than_one_run_folder(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    run_dirs = []
    for arm in ("code-score", "hybrid"):
        main(
            ["run", "--arm", arm, "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
        )
        run_dirs.append(str(next((runs_root / arm).iterdir())))
    capsys.readouterr()

    assert main(["summary", *run_dirs, "--no-html"]) == 0
    printed = capsys.readouterr().out
    assert "code-score" in printed
    assert "hybrid" in printed


def test_summary_without_html_does_not_write_a_file(config_path, tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    capsys.readouterr()
    assert main(["summary", str(runs_root), "--no-html"]) == 0
    assert not (tmp_path / "report.html").exists()


def test_summary_rejects_a_folder_that_is_not_a_run(tmp_path, capsys) -> None:
    code = main(["summary", str(tmp_path / "nothing"), "--no-html"])
    captured = capsys.readouterr()
    assert code != 0
    assert "not a run folder" in captured.err


def test_summary_needs_at_least_one_path(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["summary", "--no-html"])
    assert excinfo.value.code != 0


def test_publish_pushes_the_report_to_a_local_bare_repo(tmp_path, capsys) -> None:
    bare = tmp_path / "pages.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    report = tmp_path / "report.html"
    report.write_text("<!DOCTYPE html><p>study</p>\n", encoding="utf-8")

    code = main(["publish", "--report", str(report), "--remote", str(bare)])

    assert code == 0
    assert "published" in capsys.readouterr().out
    shown = subprocess.run(
        ["git", "--git-dir", str(bare), "show", "gh-pages:index.html"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert shown.stdout.strip() == "<!DOCTYPE html><p>study</p>"


def test_publish_reports_a_missing_report(tmp_path, capsys) -> None:
    code = main(
        ["publish", "--report", str(tmp_path / "absent.html"), "--remote", str(tmp_path / "bare.git")]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "report not found" in captured.err


SPLIT_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 1
prompts_per_step = 1

[run]
eval_every_steps = 1
eval_questions = 100
eval_subset_seed = 2026
"""


def write_split_config(tmp_path) -> Path:
    path = tmp_path / "split.toml"
    path.write_text(SPLIT_CONFIG, encoding="utf-8")
    return path


def test_build_prompts_separates_the_train_and_test_questions(tmp_path) -> None:
    config = load_config(write_split_config(tmp_path))
    train, evaluation = build_prompts(config)

    assert len(train) == 714
    assert len(evaluation) == 250
    assert {prompt.prompt_id for prompt in train}.isdisjoint(
        {prompt.prompt_id for prompt in evaluation}
    )
    split = load_split(seed=7)
    assert {prompt.prompt_id for prompt in train} == {
        prompt_id_for(question) for question in split.train
    }
    assert {prompt.prompt_id for prompt in evaluation} == {
        prompt_id_for(question) for question in split.test
    }
    assert train[0].problem.startswith("Write a Python function")


def test_a_fake_run_evaluates_the_held_out_test_split(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    assert (
        main(
            [
                "run",
                "--arm",
                "code-score",
                "--fake",
                "--config",
                str(write_split_config(tmp_path)),
                "--runs-root",
                str(runs_root),
            ]
        )
        == 0
    )
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    split = load_split(seed=7)
    train_ids = {prompt_id_for(question) for question in split.train}
    test_ids = {prompt_id_for(question) for question in split.test}

    final = payload["evaluations"][-1]
    assert payload["step"] == 1
    assert payload["count"] == 250
    assert final["kind"] == "final"
    assert final["count"] == 250
    assert set(final["prompt_ids"]) == test_ids

    checkpoints = [
        point for point in payload["evaluations"] if point["kind"] == "checkpoint"
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0]["count"] == 100
    assert len(set(checkpoints[0]["prompt_ids"])) == 100
    assert set(checkpoints[0]["prompt_ids"]) <= test_ids

    records = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # One step trains one prompt group (8 identical rollouts), so the rollout
    # prompt ids are a subset of the train ids, never a test id.
    assert {record["prompt_id"] for record in records} <= train_ids
    assert {record["prompt_id"] for record in records}.isdisjoint(test_ids)


def test_a_bad_split_stops_before_the_run_folder(
    config_path, tmp_path, monkeypatch, capsys
) -> None:
    from rlvr import cli
    from rlvr.data import DatasetError

    def fail(*args, **kwargs):
        raise DatasetError("dataset file not found")

    monkeypatch.setattr(cli, "load_split", fail)
    runs_root = tmp_path / "runs"
    code = main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "MBPP split" in captured.err
    assert not (runs_root / "code-score").exists()


RESUME_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 3
prompts_per_step = 1

[run]
checkpoint_every_steps = 2
"""


def test_resume_continues_from_the_last_checkpoint_without_duplicates(
    tmp_path, capsys
) -> None:
    runs_root = tmp_path / "runs"
    config_path = tmp_path / "resume.toml"
    config_path.write_text(RESUME_CONFIG, encoding="utf-8")

    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    before = [
        line for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(before) == 24

    assert main(
        ["run", "--arm", "code-score", "--fake", "--resume", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    capsys.readouterr()

    # Still one run folder, still 24 rollouts: step 3 was truncated then redone.
    assert len(list((runs_root / "code-score").iterdir())) == 1
    after = [
        line for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(after) == 24


def test_resume_without_a_checkpoint_fails_cleanly(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    (runs_root / "code-score").mkdir(parents=True)
    (runs_root / "code-score" / "20261112T000000Z").mkdir()
    (runs_root / "code-score" / "20261112T000000Z" / "meta.json").write_text(
        '{"arm": "code-score"}\n', encoding="utf-8"
    )

    code = main(
        ["run", "--arm", "code-score", "--fake", "--resume", "--runs-root", str(runs_root)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "no checkpoint" in captured.err


EVAL_LIMIT_CONFIG = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 7

[fake]
steps = 1
prompts_per_step = 1

[run]
eval_every_steps = 1
eval_questions = 5
eval_final_questions = 10
"""


def test_eval_final_questions_caps_the_final_eval_pool(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    config_path = tmp_path / "limit.toml"
    config_path.write_text(EVAL_LIMIT_CONFIG, encoding="utf-8")

    assert main(
        ["run", "--arm", "code-score", "--fake", "--config", str(config_path), "--runs-root", str(runs_root)]
    ) == 0
    capsys.readouterr()

    run_dir = next((runs_root / "code-score").iterdir())
    payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert payload["count"] == 10
    assert payload["evaluations"][-1]["count"] == 10


def test_build_test_runner_uses_fakes_or_docker(write_config) -> None:
    from rlvr.cli import build_test_runner
    from rlvr.docker_runner import DockerTestRunner
    from rlvr.fake import FakeTestRunner

    config = load_config(write_config())
    assert isinstance(build_test_runner(config, fake=True), FakeTestRunner)
    # The real runner is built from the merged train and test questions. No
    # Docker binary is invoked at construction, so this is safe on CPU.
    runner = build_test_runner(config, fake=False)
    assert isinstance(runner, DockerTestRunner)
