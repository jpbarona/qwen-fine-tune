"""Tests for the control file that changes a run between steps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.control import (
    ControlError,
    ControlState,
    ControlWatcher,
    parse_control,
)


def test_a_missing_file_means_do_nothing(tmp_path: Path) -> None:
    watcher = ControlWatcher(tmp_path / "control.json")
    assert watcher.poll() == ControlState()
    assert watcher.error is None


def test_the_file_is_created_empty_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    watcher = ControlWatcher(path)
    watcher.ensure_file()
    assert json.loads(path.read_text()) == {"drop": [], "tests": {}}
    assert watcher.poll().is_empty()


def test_a_drop_list_is_read(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"drop": ["mbpp-679"]}))
    state = ControlWatcher(path).poll()
    assert state.drop == frozenset({"mbpp-679"})
    assert state.tests == {}


def test_a_test_replacement_is_read_and_compiled(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps({"tests": {"mbpp-535": ["assert f(1) == 1", "assert f(2) == 2"]}})
    )
    state = ControlWatcher(path).poll()
    assert state.tests["mbpp-535"] == ("assert f(1) == 1", "assert f(2) == 2")


def test_a_test_that_does_not_compile_is_rejected() -> None:
    with pytest.raises(ControlError, match="does not compile"):
        parse_control({"tests": {"p": ["assert f(1) =="]}})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ControlError, match="unknown control keys"):
        parse_control({"drops": ["p"]})


def test_bad_shapes_are_rejected() -> None:
    for raw in ([], {"drop": "p"}, {"drop": [1]}, {"tests": []}, {"tests": {"p": []}}):
        with pytest.raises(ControlError):
            parse_control(raw)


def test_a_broken_file_keeps_the_previous_state_and_reports_the_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"drop": ["mbpp-679"]}))
    watcher = ControlWatcher(path)
    assert watcher.poll().drop == frozenset({"mbpp-679"})

    path.write_text("{not json")
    state = watcher.poll()
    assert state.drop == frozenset({"mbpp-679"})
    assert watcher.error is not None


def test_unchanged_content_is_not_reparsed(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"drop": ["a"]}))
    watcher = ControlWatcher(path)
    watcher.poll()

    calls = []
    import rlvr.control as control

    real = control.parse_control
    monkeypatch.setattr(control, "parse_control", lambda raw: calls.append(raw) or real(raw))
    watcher.poll()
    assert calls == []


def test_duplicate_drops_collapse() -> None:
    state = parse_control({"drop": ["p", "p", "q"]})
    assert state.drop == frozenset({"p", "q"})


def test_an_empty_object_is_valid() -> None:
    assert parse_control({}).is_empty()


def test_whitespace_only_test_is_rejected() -> None:
    with pytest.raises(ControlError):
        parse_control({"tests": {"p": ["   "]}})


def test_null_is_rejected() -> None:
    with pytest.raises(ControlError):
        parse_control(None)


def test_a_directory_in_place_of_the_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.mkdir()
    watcher = ControlWatcher(path)
    assert watcher.poll().is_empty()
    assert watcher.error is None


def test_the_last_good_state_survives_a_delete(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"drop": ["p"]}))
    watcher = ControlWatcher(path)
    assert watcher.poll().drop == frozenset({"p"})
    path.unlink()
    assert watcher.poll().is_empty()
