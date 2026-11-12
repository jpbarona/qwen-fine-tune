"""Tests for checkpoint directory and manifest handling. No torch."""

from __future__ import annotations

import json

import pytest

from rlvr.checkpoint import (
    CheckpointError,
    checkpoint_dir_name,
    find_latest_checkpoint,
    parse_checkpoint_step,
    read_checkpoint_manifest,
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
