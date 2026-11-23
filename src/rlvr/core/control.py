"""The control file a training run reads between steps.

A run holds its prompts, its stage pools, and the test lists in memory, so
editing the dataset while it runs changes nothing. This file is the supported
way to change a run in place. It lives in the run folder, next to `meta.json`:

    {
      "drop": ["mbpp-679", "mbpp-536"],
      "tests": {"mbpp-535": ["assert topbottom_surfacearea(10) == 314.15"]}
    }

- `drop` removes a prompt from the current stage pool, so it is not sampled
  again. The rollouts already recorded stay in the history.
- `tests` replaces the hidden test list for one prompt. From the next step the
  prompt is scored against the new list.

The file is read at the start of every step. A missing file means "do nothing".
A file that cannot be parsed is reported as an event and the previous state is
kept, so a typo cannot stop a run.

Nothing here touches the running process from the outside. The loop polls the
file itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CONTROL_NAME",
    "ControlError",
    "ControlState",
    "ControlWatcher",
    "parse_control",
]

#: File name inside a run folder.
CONTROL_NAME = "control.json"

#: The keys the file understands. Anything else is a typo, not a feature.
CONTROL_KEYS = frozenset({"drop", "tests"})


class ControlError(ValueError):
    """The control file is not the shape this module documents."""


@dataclass(frozen=True)
class ControlState:
    """What the control file asks for. Empty means do nothing."""

    drop: frozenset[str] = frozenset()
    tests: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.drop and not self.tests


def parse_control(raw: object) -> ControlState:
    """Turn decoded JSON into a `ControlState`, or raise `ControlError`."""
    if not isinstance(raw, dict):
        raise ControlError("the control file must hold a JSON object")
    unknown = set(raw) - CONTROL_KEYS
    if unknown:
        raise ControlError(
            f"unknown control keys: {', '.join(sorted(unknown))}. "
            f"expected {', '.join(sorted(CONTROL_KEYS))}"
        )

    drop_raw = raw.get("drop", [])
    if not isinstance(drop_raw, list) or any(
        not isinstance(item, str) or not item for item in drop_raw
    ):
        raise ControlError("drop must be a list of prompt id strings")

    tests_raw = raw.get("tests", {})
    if not isinstance(tests_raw, dict):
        raise ControlError("tests must be an object of prompt id to test list")
    tests: dict[str, tuple[str, ...]] = {}
    for prompt_id, entries in tests_raw.items():
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ControlError("tests keys must be prompt id strings")
        if not isinstance(entries, list) or not entries:
            raise ControlError(f"tests for {prompt_id!r} must be a non-empty list")
        if any(not isinstance(entry, str) or not entry.strip() for entry in entries):
            raise ControlError(f"tests for {prompt_id!r} must be non-empty strings")
        for entry in entries:
            try:
                compile(entry, f"{prompt_id}.py", "exec")
            except SyntaxError as error:
                raise ControlError(
                    f"test for {prompt_id!r} does not compile: {error}"
                ) from error
        tests[prompt_id] = tuple(entries)

    return ControlState(drop=frozenset(drop_raw), tests=tests)


class ControlWatcher:
    """Reads the control file when it changes, and remembers the last state.

    Polling is cheap: one `stat` per training step. A parse error leaves the
    previous state in place and is returned to the caller as an error, so the
    run can record it and carry on.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._last_text: str | None = None
        self._state = ControlState()
        self.error: str | None = None

    @property
    def state(self) -> ControlState:
        return self._state

    def ensure_file(self) -> None:
        """Create an empty control file when it is missing."""
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"drop": [], "tests": {}}\n', encoding="utf-8")
        self._last_text = None

    def poll(self) -> ControlState:
        """Return the current state, re-reading the file when it changed."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            self.error = None
            self._last_text = None
            self._state = ControlState()
            return self._state
        if text == self._last_text:
            return self._state
        self._last_text = text
        try:
            self._state = parse_control(json.loads(text))
            self.error = None
        except (json.JSONDecodeError, ControlError) as error:
            self.error = str(error)
        return self._state
