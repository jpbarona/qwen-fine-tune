"""Tests for the run folder layout, meta.json, metrics, eval, and determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.schema import FIELD_NAMES, RolloutRecord

TIMESTAMP = "20261112T000000Z"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_run_folder_is_under_arm_and_timestamp(run_fake) -> None:
    run_dir = run_fake("code-score")[0]
    assert run_dir.parent.name == "code-score"
    assert run_dir.name == TIMESTAMP
    assert run_dir.parent.parent.name == "runs"


@pytest.mark.parametrize(
    "name", ["meta.json", "metrics.jsonl", "rollouts.jsonl", "eval.json", "checkpoints"]
)
def test_run_folder_contains_the_five_expected_entries(run_fake, name: str) -> None:
    run_dir = run_fake("code-score")[0]
    assert (run_dir / name).exists(), name


def test_checkpoints_is_a_directory(run_fake) -> None:
    assert run_fake("code-score")[0].joinpath("checkpoints").is_dir()


def test_meta_json_records_reproducibility_fields(run_fake, monkeypatch) -> None:
    import rlvr.repro as repro

    monkeypatch.setattr(repro, "git_commit", lambda: "deadbeef")
    meta = read_json(run_fake("code-score")[0] / "meta.json")
    assert meta["arm"] == "code-score"
    assert meta["timestamp"] == TIMESTAMP
    assert meta["git_commit"] == "deadbeef"
    assert isinstance(meta["versions"], dict) and meta["versions"]
    assert meta["model_revision"] == "fake-revision"
    assert meta["judge_model"] == "fake-judge-model"
    assert meta["split_seed"] == 1234
    assert meta["config"]["model"]["name"] == "Qwen/Qwen3-0.6B"
    assert meta["config"]["fake"]["steps"] == 3


def test_meta_git_commit_is_null_when_git_is_unavailable(run_fake, monkeypatch) -> None:
    import rlvr.repro as repro

    monkeypatch.setattr(repro, "git_commit", lambda: None)
    meta = read_json(run_fake("code-score")[0] / "meta.json")
    assert meta["git_commit"] is None


def test_meta_json_never_contains_a_key_or_secret(run_fake) -> None:
    meta = json.dumps(read_json(run_fake("code-score")[0] / "meta.json"))
    # The trainer metadata records "max_prompt_tokens" and "max_completion_tokens",
    # so the bare substring "token" is no longer a useful secret probe. Check for
    # the markers that would actually leak a key or a secret instead.
    for banned in ("api_key", "apikey", "secret", "oc_sk_", "bearer", "authorization"):
        assert banned not in meta, banned


def test_versions_include_python_and_the_package(run_fake) -> None:
    versions = read_json(run_fake("code-score")[0] / "meta.json")["versions"]
    assert versions["python"].startswith("3.")
    assert "rlvr" in versions


def test_rollouts_jsonl_has_one_line_per_rollout(run_fake) -> None:
    # 3 steps times 8 generations per prompt group.
    run_dir, result = run_fake("code-score")
    lines = read_jsonl(run_dir / "rollouts.jsonl")
    assert result.rollouts == 24
    assert len(lines) == 24


def test_every_rollout_line_satisfies_the_schema(run_fake) -> None:
    run_dir = run_fake("code-score")[0]
    for payload in read_jsonl(run_dir / "rollouts.jsonl"):
        record = RolloutRecord.from_dict(payload)
        assert list(record.to_dict()) == list(FIELD_NAMES)


def test_metrics_jsonl_has_one_row_per_step(run_fake) -> None:
    metrics = read_jsonl(run_fake("code-score")[0] / "metrics.jsonl")
    assert [row["step"] for row in metrics] == [1, 2, 3]
    for row in metrics:
        assert row["arm"] == "code-score"
        assert isinstance(row["rollouts"], int)
        assert 0.0 <= row["mean_reward"] <= 1.0
        assert 0.0 <= row["pass_rate"] <= 1.0
        assert row["mean_entropy"] >= 0.0
        assert row["mean_completion_tokens"] >= 0.0


def test_metrics_and_rollouts_are_appended_in_step_order(run_fake) -> None:
    run_dir = run_fake("code-score")[0]
    steps = [json.loads(line)["step"] for line in (run_dir / "rollouts.jsonl").read_text().splitlines()]
    assert steps == [1] * 8 + [2] * 8 + [3] * 8


def test_eval_json_reports_pass_at_1_for_the_final_eval(run_fake) -> None:
    eval_payload = read_json(run_fake("code-score")[0] / "eval.json")
    assert eval_payload["arm"] == "code-score"
    assert eval_payload["timestamp"] == TIMESTAMP
    assert 0.0 <= eval_payload["pass_at_1"] <= 1.0
    assert eval_payload["step"] == 3
    assert eval_payload["count"] == 2
    assert [row["step"] for row in eval_payload["evaluations"]] == [3]
    assert eval_payload["evaluations"][0]["kind"] == "final"
    assert eval_payload["evaluations"][0]["prompt_ids"] == ["eval-0000", "eval-0001"]
    assert isinstance(eval_payload["evaluations"][0]["pass_at_1"], float)


def test_eval_json_records_the_checkpoint_subset_selector(run_fake) -> None:
    config = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "fake-revision"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1234

[fake]
steps = 3
prompts_per_step = 2

[run]
eval_every_steps = 2
eval_questions = 2
eval_subset_seed = 5
"""
    eval_payload = read_json(run_fake("code-score", config_text=config)[0] / "eval.json")
    assert eval_payload["subset"] == {
        "method": "seeded_sample",
        "seed": 5,
        "requested": 2,
        "selected": 2,
        "pool_size": 2,
    }
    assert [row["kind"] for row in eval_payload["evaluations"]] == [
        "checkpoint",
        "final",
    ]
    assert [row["step"] for row in eval_payload["evaluations"]] == [2, 3]


def test_eval_json_is_written_atomically(tmp_path, monkeypatch) -> None:
    from rlvr.runs import RunStore

    store = RunStore(tmp_path / "runs", "code-score", timestamp=TIMESTAMP)
    store.write_eval({"pass_at_1": 0.5})
    before = store.eval_path.read_text(encoding="utf-8")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("rlvr.runs.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        store.write_eval({"pass_at_1": 0.9})

    assert store.eval_path.read_text(encoding="utf-8") == before
    assert [path.name for path in store.run_dir.iterdir() if path.name.endswith(".tmp")] == []


def test_meta_json_is_written_atomically(tmp_path, monkeypatch) -> None:
    from rlvr.runs import RunStore

    store = RunStore(tmp_path / "runs", "hybrid", timestamp=TIMESTAMP)
    store.write_meta({"arm": "hybrid"})
    before = store.meta_path.read_text(encoding="utf-8")
    monkeypatch.setattr("rlvr.runs.os.replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.write_meta({"arm": "other"})
    assert store.meta_path.read_text(encoding="utf-8") == before


def test_two_fake_runs_produce_the_same_rollouts(run_fake) -> None:
    first = (run_fake("code-score", runs_root=Path("runs_a"))[0] / "rollouts.jsonl").read_bytes()
    second = (run_fake("code-score", runs_root=Path("runs_b"))[0] / "rollouts.jsonl").read_bytes()
    assert first == second


def test_two_fake_runs_produce_the_same_metrics_and_eval(run_fake) -> None:
    a = run_fake("code-score", runs_root=Path("runs_a"))[0]
    b = run_fake("code-score", runs_root=Path("runs_b"))[0]
    assert (a / "metrics.jsonl").read_bytes() == (b / "metrics.jsonl").read_bytes()
    assert (a / "eval.json").read_bytes() == (b / "eval.json").read_bytes()


def test_a_second_run_creates_a_new_folder(run_fake) -> None:
    first = run_fake("code-score")[0]
    second = run_fake("code-score", timestamp="20261201T101010Z")[0]
    assert first != second
    assert second.name == "20261201T101010Z"


def test_a_store_never_overwrites_an_existing_run_folder(tmp_path) -> None:
    from rlvr.runs import RunStore

    RunStore(tmp_path / "runs", "hybrid", timestamp=TIMESTAMP)
    with pytest.raises(FileExistsError, match="already exists"):
        RunStore(tmp_path / "runs", "hybrid", timestamp=TIMESTAMP)


def test_timestamp_must_be_one_path_segment(tmp_path) -> None:
    from rlvr.runs import RunStore

    for bad in ("a/b", "", ".."):
        with pytest.raises(ValueError, match="path segment"):
            RunStore(tmp_path / "runs", "hybrid", timestamp=bad)


def test_default_timestamp_is_utc_and_path_safe() -> None:
    import re

    from rlvr.runs import utc_timestamp

    assert re.fullmatch(r"\d{8}T\d{6}Z", utc_timestamp())
