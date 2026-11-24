from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Any, Callable, Iterator

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class StopRequest(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum

    @property
    def name(self) -> str:
        return signal.Signals(self.signum).name

    @property
    def exit_code(self) -> int:
        return 128 + self.signum


def mirror_status(*, stopped: bool, failed: bool) -> str:
    if stopped:
        return "KILLED"
    if failed:
        return "FAILED"
    return "FINISHED"


def _handle_stop(signum: int, frame: FrameType | None) -> None:
    signal.signal(signum, signal.SIG_DFL)
    raise StopRequest(signum)


def _install(
    handle: Callable[[int, FrameType | None], None],
) -> dict[int, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    for signum in STOP_SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    return previous


@contextmanager
def stop_on_signals() -> Iterator[None]:
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
