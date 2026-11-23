"""Tests for checkpoint directory and manifest handling. No torch."""

from __future__ import annotations

import json

import pytest

from rlvr.core.checkpoint import (
    CheckpointError,
    checkpoint_dir_name,
    find_latest_checkpoint,
    parse_checkpoint_step,
    prune_checkpoints,
    read_checkpoint_manifest,
    select_checkpoints_to_keep,
    write_checkpoint_manifest,
)


def test_dir_name_is_zero_padded_eight_digits() -> None:
    assert checkpoint_dir_name(1) == "step-00000001"
    assert checkpoint_dir_name(50) == "step-00000050"
    assert checkpoint_dir_name(3570) == "step-00003570"


def test_dir_name_rejects_bad_steps() -> None:
    for bad in (0, -1, 1.5, "1", True):
        with pytest.raises(ValueError):
            checkpoint_dir_name(bad)  # type: ignore[arg-type]


def test_parse_step_round_trips() -> None:
    for step in (1, 50, 99999999):
        assert parse_checkpoint_step(checkpoint_dir_name(step)) == step


def test_parse_step_rejects_non_checkpoint_names() -> None:
    for name in ("step-50", "step-0000000", "step-00000000", "checkpoint", "step-abcdefgh"):
        assert parse_checkpoint_step(name) is None


def test_manifest_round_trips(tmp_path) -> None:
    directory = tmp_path / "step-00000050"
    write_checkpoint_manifest(directory, step=50, arm="hybrid", timestamp="20261112T000000Z")
    payload = read_checkpoint_manifest(directory)
    assert payload["step"] == 50
    assert payload["arm"] == "hybrid"
    assert payload["timestamp"] == "20261112T000000Z"


def test_manifest_carries_the_prompt_cursor_and_batch_state(tmp_path) -> None:
    directory = tmp_path / "step-00000050"
    write_checkpoint_manifest(
        directory,
        step=50,
        arm="code-score",
        timestamp="t",
        prompt_cursor=125,
        batch={"current": 4, "largest_fit": 3, "settled": False},
        stage={"index": 1, "step": 4, "pool": ["mbpp-0002", "mbpp-0003"]},
    )
    payload = read_checkpoint_manifest(directory)
    assert payload["prompt_cursor"] == 125
    assert payload["batch"] == {"current": 4, "largest_fit": 3, "settled": False}
    assert payload["stage"] == {
        "index": 1,
        "step": 4,
        "pool": ["mbpp-0002", "mbpp-0003"],
    }


def test_manifest_omits_the_new_fields_when_not_given(tmp_path) -> None:
    directory = tmp_path / "step-00000050"
    write_checkpoint_manifest(directory, step=50, arm="code-score", timestamp="t")
    payload = read_checkpoint_manifest(directory)
    assert "prompt_cursor" not in payload
    assert "batch" not in payload
    assert "stage" not in payload


def test_manifest_is_json_with_trailing_newline(tmp_path) -> None:
    directory = tmp_path / "step-00000001"
    write_checkpoint_manifest(directory, step=1, arm="code-score", timestamp="t")
    text = (directory / "checkpoint.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["step"] == 1


def test_manifest_rejects_a_missing_file(tmp_path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        read_checkpoint_manifest(tmp_path / "absent")


def test_manifest_rejects_a_bad_step(tmp_path) -> None:
    directory = tmp_path / "step-00000050"
    directory.mkdir()
    (directory / "checkpoint.json").write_text(
        json.dumps({"step": "fifty", "arm": "hybrid"}), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="step"):
        read_checkpoint_manifest(directory)


def test_manifest_rejects_a_bad_arm(tmp_path) -> None:
    directory = tmp_path / "step-00000050"
    directory.mkdir()
    (directory / "checkpoint.json").write_text(
        json.dumps({"step": 50, "arm": ""}), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="arm"):
        read_checkpoint_manifest(directory)


def test_find_latest_returns_none_for_an_empty_dir(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    assert find_latest_checkpoint(checkpoints) is None


def test_find_latest_returns_none_for_a_missing_dir(tmp_path) -> None:
    assert find_latest_checkpoint(tmp_path / "absent") is None


def test_find_latest_picks_the_highest_step(tmp_path) -> None:
    for step in (50, 100, 150):
        write_checkpoint_manifest(
            tmp_path / checkpoint_dir_name(step),
            step=step,
            arm="hybrid",
            timestamp="t",
        )
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.step == 150


def test_find_latest_skips_corrupt_and_non_step_dirs(tmp_path) -> None:
    write_checkpoint_manifest(
        tmp_path / checkpoint_dir_name(50), step=50, arm="hybrid", timestamp="t"
    )
    corrupt = tmp_path / "step-00000060"
    corrupt.mkdir()
    (corrupt / "checkpoint.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "not-a-checkpoint").mkdir()
    (tmp_path / "step-0000000").mkdir()
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.step == 50


def test_select_keeps_every_checkpoint_when_under_the_total() -> None:
    assert select_checkpoints_to_keep([20, 40, 60]) == [20, 40, 60]


def test_latest_skips_a_manifest_that_names_a_different_step(tmp_path) -> None:
    for directory_step, manifest_step in ((10, 10), (20, 19)):
        write_checkpoint_manifest(
            tmp_path / checkpoint_dir_name(directory_step),
            step=manifest_step, arm="code-score", timestamp="t",
        )
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.step == 10


def test_select_keeps_fifteen_of_forty_at_doubling_gaps() -> None:
    steps = [20 * index for index in range(1, 41)]
    kept = select_checkpoints_to_keep(steps)
    assert len(kept) == 15
    # The five newest are consecutive. Older rungs sit at doubling gaps, and
    # the oldest checkpoints fill the remaining slots.
    assert kept == [20, 40, 60, 80, 100, 120, 140, 440, 600, 680, 720, 740, 760, 780, 800]
    assert kept[-5:] == [720, 740, 760, 780, 800]


def test_select_sorts_and_dedupes_before_choosing() -> None:
    kept = select_checkpoints_to_keep([60, 20, 40, 40], recent=2, total=3)
    assert kept == [20, 40, 60]


def test_select_on_an_empty_sequence_keeps_nothing() -> None:
    assert select_checkpoints_to_keep([]) == []


def test_select_rejects_a_bad_window() -> None:
    for recent, total in ((0, 10), (5, 0), (6, 5)):
        with pytest.raises(ValueError):
            select_checkpoints_to_keep([20, 40], recent=recent, total=total)


def test_prune_removes_only_unkept_step_directories(tmp_path) -> None:
    for step in (20, 40, 60, 80):
        write_checkpoint_manifest(
            tmp_path / checkpoint_dir_name(step), step=step, arm="hybrid", timestamp="t"
        )
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
    (tmp_path / "other").mkdir()
    (tmp_path / "step-50").mkdir()

    deleted = prune_checkpoints(tmp_path, [40, 80])

    assert sorted(path.name for path in deleted) == [
        "step-00000020",
        "step-00000060",
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "notes.txt",
        "other",
        "step-00000040",
        "step-00000080",
        "step-50",
    ]


def test_prune_leaves_a_file_named_like_a_step_alone(tmp_path) -> None:
    marker = tmp_path / checkpoint_dir_name(20)
    marker.write_text("not a checkpoint", encoding="utf-8")
    assert prune_checkpoints(tmp_path, []) == []
    assert marker.read_text(encoding="utf-8") == "not a checkpoint"


def test_prune_on_a_missing_directory_deletes_nothing(tmp_path) -> None:
    assert prune_checkpoints(tmp_path / "absent", [20]) == []


def test_prune_with_every_step_kept_deletes_nothing(tmp_path) -> None:
    for step in (20, 40):
        (tmp_path / checkpoint_dir_name(step)).mkdir()
    assert prune_checkpoints(tmp_path, [20, 40]) == []
    assert len(list(tmp_path.iterdir())) == 2
