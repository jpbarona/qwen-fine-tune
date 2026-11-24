from __future__ import annotations

import signal

import pytest

from rlvr.app.stop import (
    StopRequest,
    _handle_stop,
    mirror_status,
    stop_on_signals,
)


def test_a_stop_request_bypasses_except_exception() -> None:
    assert issubclass(StopRequest, BaseException)
    assert not issubclass(StopRequest, Exception)


@pytest.mark.parametrize(
    ("signum", "exit_code", "name"),
    [
        (signal.SIGTERM, 143, "SIGTERM"),
        (signal.SIGINT, 130, "SIGINT"),
    ],
)
def test_a_stop_request_names_its_signal(signum, exit_code, name) -> None:
    request = StopRequest(signum)

    assert request.signum == signum
    assert request.exit_code == exit_code
    assert request.name == name


def test_the_handler_sets_the_default_action_before_it_raises(monkeypatch) -> None:
    calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        signal, "signal", lambda signum, handler: calls.append((signum, handler))
    )

    with pytest.raises(StopRequest) as caught:
        _handle_stop(signal.SIGTERM, None)

    assert calls == [(signal.SIGTERM, signal.SIG_DFL)]
    assert caught.value.signum == signal.SIGTERM


def test_the_block_installs_both_handlers_and_restores_them(monkeypatch) -> None:
    installed: dict[int, object] = {}
    previous = {signal.SIGTERM: "term-before", signal.SIGINT: "int-before"}
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous[signum])
    monkeypatch.setattr(
        signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler)
    )

    with stop_on_signals():
        assert callable(installed[signal.SIGTERM])
        assert callable(installed[signal.SIGINT])

    assert installed == previous


def test_a_fired_signal_keeps_the_default_action(monkeypatch) -> None:
    installed: dict[int, object] = {}
    previous = {signal.SIGTERM: "term-before", signal.SIGINT: "int-before"}
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous[signum])
    monkeypatch.setattr(
        signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler)
    )

    with pytest.raises(StopRequest):
        with stop_on_signals():
            installed[signal.SIGTERM](signal.SIGTERM, None)

    assert installed[signal.SIGTERM] is signal.SIG_DFL
    assert installed[signal.SIGINT] == "int-before"


@pytest.mark.parametrize(
    ("stopped", "failed", "status"),
    [
        (True, False, "KILLED"),
        (True, True, "KILLED"),
        (False, True, "FAILED"),
        (False, False, "FINISHED"),
    ],
)
def test_the_mirror_status_follows_the_end_of_the_run(stopped, failed, status) -> None:
    assert mirror_status(stopped=stopped, failed=failed) == status
