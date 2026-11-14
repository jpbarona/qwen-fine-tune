"""The synthetic very easy question set that gives the first stage its material.

The committed file `data/synthetic_easy.jsonl` uses the MBPP row shape plus the
reference solution:

    task_id, text, code, test_list, test_setup_code

`task_id` starts at 10001, far above the MBPP range of 1 to 974, so a synthetic
id can never collide with an MBPP id. The prompt id carries the `synthetic-`
prefix, so the two sets are also distinct by id string.

Every question must have at least two tests and a reference solution of at most
19 lines. The loader enforces both, so a hard question cannot slip into the
stage that is meant to be solvable. The code is standard library only. A test
checks that, and executes every reference solution against its own tests with
plain Python. No model, network, or Docker call is needed for that check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .data import MBPPQuestion

__all__ = [
    "DEFAULT_SYNTHETIC_PATH",
    "MAX_REFERENCE_LINES",
    "MIN_TESTS",
    "SYNTHETIC_SOURCE_SPLIT",
    "SYNTHETIC_TASK_ID_START",
    "SyntheticError",
    "SyntheticQuestion",
    "load_synthetic_questions",
]

#: Repo-relative path of the committed synthetic set.
DEFAULT_SYNTHETIC_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "synthetic_easy.jsonl"
)

#: First synthetic task id. MBPP ids stop at 974, so this cannot collide.
SYNTHETIC_TASK_ID_START = 10001

#: Where synthetic rows say they came from. MBPP rows never use this value.
SYNTHETIC_SOURCE_SPLIT = "synthetic"

#: The question must be solvable in under 20 lines, so 19 is the maximum.
MAX_REFERENCE_LINES = 19

#: A synthetic question needs at least two tests to be worth running.
MIN_TESTS = 2


class SyntheticError(ValueError):
    """The synthetic question file is missing or invalid."""


@dataclass(frozen=True)
class SyntheticQuestion:
    """One synthetic question with its reference solution and tests."""

    task_id: int
    text: str
    code: str
    test_list: tuple[str, ...]
    test_setup_code: str = ""

    @property
    def prompt_id(self) -> str:
        return f"synthetic-{self.task_id}"

    def as_question(self) -> MBPPQuestion:
        """The row the runner and the prompt writer both understand."""
        return MBPPQuestion(
            task_id=self.task_id,
            text=self.text,
            test_list=self.test_list,
            test_setup_code=self.test_setup_code,
            source_split=SYNTHETIC_SOURCE_SPLIT,
            prompt_id=self.prompt_id,
        )


def _line_count(code: str) -> int:
    return len(code.strip().splitlines())


def _question_from_row(row: object, line_number: int) -> SyntheticQuestion:
    if not isinstance(row, dict):
        raise SyntheticError(
            f"synthetic line {line_number}: row must be an object"
        )

    task_id = row.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise SyntheticError(
            f"synthetic line {line_number}: 'task_id' must be an integer"
        )
    if task_id < SYNTHETIC_TASK_ID_START:
        raise SyntheticError(
            f"synthetic line {line_number}: task id {task_id} collides with the "
            f"MBPP range, it must be at least {SYNTHETIC_TASK_ID_START}"
        )

    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SyntheticError(
            f"synthetic line {line_number}: 'text' must be a non-empty string"
        )

    code = row.get("code")
    if not isinstance(code, str) or not code.strip():
        raise SyntheticError(
            f"synthetic line {line_number}: 'code' must be a non-empty string"
        )
    if _line_count(code) > MAX_REFERENCE_LINES:
        raise SyntheticError(
            f"synthetic line {line_number}: reference solution for task "
            f"{task_id} has {_line_count(code)} lines, the limit is "
            f"{MAX_REFERENCE_LINES}"
        )

    tests = row.get("test_list")
    if (
        not isinstance(tests, list)
        or not all(isinstance(test, str) and test.strip() for test in tests)
        or len(tests) < MIN_TESTS
    ):
        raise SyntheticError(
            f"synthetic line {line_number}: 'test_list' must hold at least "
            f"{MIN_TESTS} non-empty test strings"
        )

    setup = row.get("test_setup_code", "")
    if not isinstance(setup, str):
        raise SyntheticError(
            f"synthetic line {line_number}: 'test_setup_code' must be a string"
        )

    return SyntheticQuestion(
        task_id=task_id,
        text=text,
        code=code,
        test_list=tuple(tests),
        test_setup_code=setup,
    )


def load_synthetic_questions(
    path: str | Path = DEFAULT_SYNTHETIC_PATH,
) -> tuple[SyntheticQuestion, ...]:
    """Read the synthetic file in file order.

    Raises `SyntheticError` on a missing file, a bad row, a duplicate id, a
    reference solution over the line limit, or fewer than two tests.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise SyntheticError(f"synthetic file not found: {file_path}")

    questions: list[SyntheticQuestion] = []
    seen: set[int] = set()
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SyntheticError(
            f"cannot read synthetic file {file_path}: {error}"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SyntheticError(
                f"synthetic line {line_number}: not valid JSON: {error}"
            ) from error
        question = _question_from_row(row, line_number)
        if question.task_id in seen:
            raise SyntheticError(
                f"synthetic line {line_number}: duplicate task_id "
                f"{question.task_id}"
            )
        seen.add(question.task_id)
        questions.append(question)

    return tuple(questions)
