"""Difficulty screening over a question set.

The screen samples the base model on each question and records how many samples
passed every task test. It is a measurement tool: it never trains, never builds
an optimizer, never writes a checkpoint, and never writes into a run folder.

The output is a caller-supplied directory with two files:

    rows.jsonl     one row per question: id, samples, passed, pass rate
    summary.json   the pass rate histogram and the zero, partial, full counts

Rows are appended and flushed as each question finishes, so a crash costs at
most one question. A restart reads the rows already written and skips those
question ids. A torn final line, left by a write that did not finish, is dropped
and removed, so the restart appends on a clean line.

The screen samples with thinking off, at the completion cap the trainer uses.
The point is to measure the same regime the curriculum will train in.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .experiment import Prompt
from .reward import all_tests_passed

__all__ = [
    "DEFAULT_SCREEN_THINKING",
    "HISTOGRAM_LABELS",
    "ROWS_NAME",
    "SUMMARY_NAME",
    "ScreenError",
    "ScreenResult",
    "ScreenRow",
    "ScreenStore",
    "screen",
    "summarize",
]

#: Name of the per-question table inside the output directory.
ROWS_NAME = "rows.jsonl"

#: Name of the summary file inside the output directory.
SUMMARY_NAME = "summary.json"

#: The screen measures the regime the trainer will use, so thinking is off.
DEFAULT_SCREEN_THINKING = False

#: Histogram bucket labels, lowest first. One bucket per tenth of pass rate.
HISTOGRAM_LABELS: tuple[str, ...] = tuple(
    f"{index / 10:.1f}" for index in range(11)
)


class ScreenError(ValueError):
    """The screen request or an existing output table is invalid."""


class ScreenGenerator(Protocol):
    """The part of the generator seam the screen uses.

    `thinking` is explicit because the screen runs with it off while training
    runs with it on. The fake and Qwen generators both accept the flag.
    """

    def generate(
        self,
        prompt: Prompt,
        *,
        n: int,
        temperature: float,
        thinking: bool = True,
    ) -> list[Any]: ...


@dataclass(frozen=True)
class ScreenRow:
    """One question and its sample result."""

    prompt_id: str
    samples: int
    passed: int
    pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "samples": self.samples,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
        }


@dataclass(frozen=True)
class ScreenResult:
    """What one screen produced."""

    directory: Path
    rows: tuple[ScreenRow, ...]
    resumed: int
    summary: Mapping[str, Any]


def _row_from_payload(payload: object, line_number: int) -> ScreenRow:
    """Validate one row read back from disk. A bad row is a loud error."""
    if not isinstance(payload, dict):
        raise ScreenError(f"screen rows line {line_number}: row must be an object")

    prompt_id = payload.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ScreenError(
            f"screen rows line {line_number}: 'prompt_id' must be a non-empty string"
        )

    samples = payload.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ScreenError(
            f"screen rows line {line_number}: 'samples' must be an integer of at least 1"
        )

    passed = payload.get("passed")
    if (
        isinstance(passed, bool)
        or not isinstance(passed, int)
        or not 0 <= passed <= samples
    ):
        raise ScreenError(
            f"screen rows line {line_number}: 'passed' must be an integer "
            f"between 0 and {samples}"
        )

    pass_rate = payload.get("pass_rate")
    if isinstance(pass_rate, bool) or not isinstance(pass_rate, (int, float)):
        raise ScreenError(
            f"screen rows line {line_number}: 'pass_rate' must be a number"
        )
    expected = passed / samples
    if not math.isfinite(float(pass_rate)) or not math.isclose(
        float(pass_rate), expected, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ScreenError(
            f"screen rows line {line_number}: 'pass_rate' must be passed / samples, "
            f"got {pass_rate!r} for {passed} of {samples}"
        )

    return ScreenRow(
        prompt_id=prompt_id,
        samples=samples,
        passed=passed,
        pass_rate=expected,
    )


def _bucket_index(row: ScreenRow) -> int:
    """The tenth-of-a-rate bucket for a row, worked out from whole counts.

    Using `passed * 10 // samples` avoids the float rounding that a tenth of a
    percent boundary would otherwise hit, for example 7 of 10 samples.
    """
    return min((row.passed * 10) // row.samples, 10)


def summarize(
    rows: Sequence[ScreenRow], *, samples_per_question: int
) -> dict[str, Any]:
    """The summary record for a set of rows.

    `pass_rate_histogram` has one entry per tenth from `0.0` to `1.0`, so every
    bucket is present even when it is empty. `zero_pass`, `partial_pass` and
    `full_pass` count questions. A question is partial when some but not all of
    its samples fully passed.
    """
    if isinstance(samples_per_question, bool) or not isinstance(
        samples_per_question, int
    ):
        raise ScreenError(
            f"samples_per_question must be an integer, got {samples_per_question!r}"
        )
    if samples_per_question < 1:
        raise ScreenError(
            f"samples_per_question must be at least 1, got {samples_per_question}"
        )

    counts = {label: 0 for label in HISTOGRAM_LABELS}
    zero = partial = full = 0
    for row in rows:
        if row.samples != samples_per_question:
            raise ScreenError(
                f"question {row.prompt_id!r} used {row.samples} samples, "
                f"not {samples_per_question}"
            )
        counts[HISTOGRAM_LABELS[_bucket_index(row)]] += 1
        if row.passed == 0:
            zero += 1
        elif row.passed == row.samples:
            full += 1
        else:
            partial += 1

    return {
        "questions": len(rows),
        "samples_per_question": samples_per_question,
        "pass_rate_histogram": counts,
        "zero_pass": zero,
        "partial_pass": partial,
        "full_pass": full,
    }


class ScreenStore:
    """Append rows to and read rows from a screen output directory.

    The directory holds only `rows.jsonl` and `summary.json`. Nothing here
    knows about run folders, checkpoints, or optimizer state.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.directory.is_dir():
            raise ScreenError(
                f"screen output path is not a directory: {self.directory}"
            )

    @property
    def rows_path(self) -> Path:
        return self.directory / ROWS_NAME

    @property
    def summary_path(self) -> Path:
        return self.directory / SUMMARY_NAME

    def read_rows(self) -> list[ScreenRow]:
        """Rows already written, in file order.

        A torn final line is dropped and the file is rewritten without it, so
        the next append starts on a clean line. A bad line anywhere else, or a
        bad final line that was terminated with a newline, raises `ScreenError`
        because that is corruption, not an interrupted write.
        """
        if not self.rows_path.exists():
            return []
        try:
            text = self.rows_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ScreenError(f"cannot read screen rows: {error}") from error
        if not text:
            return []

        lines = text.splitlines()
        terminated = text.endswith("\n")
        rows: list[ScreenRow] = []
        seen: set[str] = set()
        torn = False
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                is_torn_tail = line_number == len(lines) and not terminated
                if is_torn_tail:
                    torn = True
                    break
                raise ScreenError(
                    f"screen rows line {line_number}: not valid JSON: {error}"
                ) from error

            row = _row_from_payload(payload, line_number)
            if row.prompt_id in seen:
                raise ScreenError(
                    f"screen rows line {line_number}: duplicate question id "
                    f"{row.prompt_id!r}"
                )
            seen.add(row.prompt_id)
            rows.append(row)

        if torn:
            self._rewrite_rows(rows)
        return rows

    def append_row(self, row: ScreenRow) -> None:
        """Append one row and flush it, so a crash keeps every finished row."""
        line = json.dumps(
            row.to_dict(), ensure_ascii=False, allow_nan=False
        )
        with self.rows_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        """Replace `summary.json` atomically."""
        _write_atomic(
            self.summary_path,
            json.dumps(dict(summary), indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
        )

    def _rewrite_rows(self, rows: Sequence[ScreenRow]) -> None:
        text = "".join(
            json.dumps(row.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        )
        _write_atomic(self.rows_path, text)


def _write_atomic(path: Path, text: str) -> None:
    """Write `text` through a temporary file in the same folder."""
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _checked_prompts(prompts: Sequence[Prompt]) -> list[Prompt]:
    checked: list[Prompt] = []
    seen: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, Prompt):
            raise ScreenError("every screened question must be a Prompt")
        if not isinstance(prompt.prompt_id, str) or not prompt.prompt_id.strip():
            raise ScreenError("every screened question must have a non-empty id")
        if prompt.prompt_id in seen:
            raise ScreenError(f"duplicate question id {prompt.prompt_id!r}")
        seen.add(prompt.prompt_id)
        checked.append(prompt)
    return checked


def screen(
    prompts: Sequence[Prompt],
    *,
    samples: int,
    directory: str | Path,
    generator: ScreenGenerator,
    test_runner: Any,
    temperature: float,
    thinking: bool = DEFAULT_SCREEN_THINKING,
) -> ScreenResult:
    """Sample every question and write the row table and summary.

    Questions already present in `rows.jsonl` are skipped, so an interrupted
    screen continues where it stopped. Existing rows must use the same sample
    count and must belong to the requested question set, so two different
    screens cannot be mixed in one directory by accident.

    A sample counts as passed only when every task test passed. The row keeps
    the pass count and the rate. The summary is rewritten after the rows, so it
    always describes every row on disk.
    """
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise ScreenError(f"samples must be an integer, got {samples!r}")
    if samples < 1:
        raise ScreenError(f"samples must be at least 1, got {samples}")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ScreenError(f"temperature must be positive, got {temperature!r}")

    checked = _checked_prompts(prompts)
    if not checked:
        raise ScreenError("the screen needs at least one question")

    store = ScreenStore(directory)
    existing = store.read_rows()

    requested_ids = {prompt.prompt_id for prompt in checked}
    done: dict[str, ScreenRow] = {}
    for row in existing:
        if row.prompt_id not in requested_ids:
            raise ScreenError(
                f"existing screen row {row.prompt_id!r} is not in this question set"
            )
        if row.samples != samples:
            raise ScreenError(
                f"existing screen row {row.prompt_id!r} used {row.samples} samples, "
                f"not {samples}"
            )
        done[row.prompt_id] = row

    for prompt in checked:
        if prompt.prompt_id in done:
            continue
        generations = generator.generate(
            prompt,
            n=samples,
            temperature=temperature,
            thinking=thinking,
        )
        if len(generations) != samples:
            raise ScreenError(
                f"generator returned {len(generations)} samples for "
                f"{prompt.prompt_id!r}, expected {samples}"
            )
        passed = 0
        for generation in generations:
            outcome = test_runner.run(prompt, generation.code)
            if all_tests_passed(outcome.tests_passed, outcome.tests_total):
                passed += 1
        row = ScreenRow(
            prompt_id=prompt.prompt_id,
            samples=samples,
            passed=passed,
            pass_rate=passed / samples,
        )
        store.append_row(row)
        done[row.prompt_id] = row

    rows = tuple(done[prompt.prompt_id] for prompt in checked)
    summary = summarize(rows, samples_per_question=samples)
    store.write_summary(summary)
    return ScreenResult(
        directory=store.directory,
        rows=rows,
        resumed=len(existing),
        summary=summary,
    )
