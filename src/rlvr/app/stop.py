"""Cooperative stop on the signals an operator sends.

`docker stop` sends SIGTERM and Ctrl-C sends SIGINT. Python's default action
for SIGTERM ends the process at once, so the uploader never runs its final
scan and MLflow keeps the run RUNNING. The handler here raises `StopRequest`
instead. It is a `BaseException`, so an `except Exception` around a step
cannot swallow the stop.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Any, Callable, Iterator

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class StopRequest(BaseException):
    """A signal asked the run to stop."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum

    @property
    def name(self) -> str:
        """The signal name, such as SIGTERM."""
        return signal.Signals(self.signum).name

    @property
    def exit_code(self) -> int:
        """The shell exit code, 143 for SIGTERM and 130 for SIGINT."""
        return 128 + self.signum


def mirror_status(*, stopped: bool, failed: bool) -> str:
    """The MLflow status for the end of one run.

    A signal stop is KILLED, an exception is FAILED, and a normal end is
    FINISHED. A stop wins over a failure, because the signal is the reason the
    process left the training loop.
    """
    if stopped:
        return "KILLED"
    if failed:
        return "FAILED"
    return "FINISHED"


def _handle_stop(signum: int, frame: FrameType | None) -> None:
    """Ask for a clean stop and make the next signal kill at once."""
    signal.signal(signum, signal.SIG_DFL)
    raise StopRequest(signum)


def _install(
    handle: Callable[[int, FrameType | None], None],
) -> dict[int, Any]:
    """Install `handle` and return the handlers it replaced.

    `signal.signal` works only on the main thread. A call from another thread
    leaves the default action in place instead of failing the run at startup.
    """
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    for signum in STOP_SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    return previous


@contextmanager
def stop_on_signals() -> Iterator[None]:
    """Turn SIGTERM and SIGINT into `StopRequest` for the body of this block.

    The previous handlers come back when the block ends, except for a signal
    that fired. That one stays at the default action, so a second signal kills
    the process even while the shutdown path runs.
    """
    fired: set[int] = set()

    def handle(signum: int, frame: FrameType | None) -> None:
        fired.add(signum)
        _handle_stop(signum, frame)

    previous = _install(handle)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            if signum not in fired:
                signal.signal(signum, handler)
