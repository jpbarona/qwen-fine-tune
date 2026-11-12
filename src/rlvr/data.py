"""The MBPP dataset, the fixed train/test split, and the prompt format.

The committed file `data/mbpp_full.jsonl` holds the MBPP `full` configuration:
374 train, 90 validation, and 500 test rows, 964 in total. The 10 few-shot
prompt rows from the same configuration are not part of the file. They are
skipped by the loader if a raw export still carries them.

The split is a pure function of the questions and the seed. It sorts by task id
first, so the caller's row order cannot change the result. Every arm calls
`load_split` or `split_questions` with the same seed from config, so every arm
sees the same 714 train and 250 test questions.

The prompt format is one function, `format_prompt`, used by every arm through
`prompt_for_question`. It is deliberately short: the problem text and a request
for a Python function. The tests never go into the prompt.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .experiment import Prompt

__all__ = [
    "DEFAULT_DATA_PATH",
    "PROMPT_TEMPLATE",
    "TEST_SIZE",
    "TOTAL_QUESTIONS",
    "TRAIN_SIZE",
    "DatasetError",
    "DatasetSplit",
    "MBPPQuestion",
    "format_prompt",
    "load_dataset",
    "load_split",
    "prompt_for_question",
    "prompt_id_for",
    "prompts_for_questions",
    "split_questions",
]

#: Repo-relative path of the committed offline dataset.
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "mbpp_full.jsonl"

#: Counts for the full MBPP configuration.
TOTAL_QUESTIONS = 964
TRAIN_SIZE = 714
TEST_SIZE = 250

#: Splits that belong in the merged dataset. `prompt` holds the 10 few-shot
#: examples and is always skipped.
SOURCE_SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: The single prompt format used by every arm. The problem text is inserted
#: verbatim. Do not add tests, hints, or example solutions here.
PROMPT_TEMPLATE = (
    "Write a Python function that solves the problem below.\n\n"
    "Problem: {text}\n\n"
    "Return only the Python code for the function."
)


class DatasetError(ValueError):
    """The dataset file or split request is invalid."""


@dataclass(frozen=True)
class MBPPQuestion:
    """One MBPP row. The tests stay on the runner side, never in the prompt."""

    task_id: int
    text: str
    test_list: tuple[str, ...]
    test_setup_code: str = ""
    source_split: str = ""


@dataclass(frozen=True)
class DatasetSplit:
    """The fixed train and test cut. Every arm reads this same object."""

    train: tuple[MBPPQuestion, ...]
    test: tuple[MBPPQuestion, ...]


def _question_from_row(row: object, line_number: int) -> MBPPQuestion | None:
    if not isinstance(row, dict):
        raise DatasetError(f"line {line_number}: row must be an object")

    source_split = row.get("source_split", "")
    if not isinstance(source_split, str):
        raise DatasetError(f"line {line_number}: 'source_split' must be a string")
    if source_split == "prompt":
        return None
    if source_split not in SOURCE_SPLITS:
        raise DatasetError(
            f"line {line_number}: unknown source_split {source_split!r}, "
            f"expected one of {', '.join(SOURCE_SPLITS)} or 'prompt'"
        )

    task_id = row.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise DatasetError(f"line {line_number}: 'task_id' must be an integer")

    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise DatasetError(f"line {line_number}: 'text' must be a non-empty string")

    tests = row.get("test_list")
    if not isinstance(tests, list) or not all(isinstance(test, str) for test in tests):
        raise DatasetError(f"line {line_number}: 'test_list' must be a list of strings")

    setup = row.get("test_setup_code", "")
    if not isinstance(setup, str):
        raise DatasetError(f"line {line_number}: 'test_setup_code' must be a string")

    return MBPPQuestion(
        task_id=task_id,
        text=text,
        test_list=tuple(tests),
        test_setup_code=setup,
        source_split=source_split,
    )


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> tuple[MBPPQuestion, ...]:
    """Read the merged dataset in train, validation, test order.

    Raises DatasetError on a missing file, a bad row, or a duplicate task id.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetError(f"dataset file not found: {file_path}")

    questions: list[MBPPQuestion] = []
    seen: set[int] = set()
    ordering = {name: index for index, name in enumerate(SOURCE_SPLITS)}
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(
                        f"line {line_number}: not valid JSON: {error}"
                    ) from error
                question = _question_from_row(row, line_number)
                if question is None:
                    continue
                if question.task_id in seen:
                    raise DatasetError(
                        f"line {line_number}: duplicate task_id {question.task_id}"
                    )
                seen.add(question.task_id)
                questions.append(question)
    except OSError as error:
        raise DatasetError(f"cannot read dataset file {file_path}: {error}") from error

    questions.sort(key=lambda question: (ordering[question.source_split], question.task_id))
    return tuple(questions)


def split_questions(
    questions: tuple[MBPPQuestion, ...] | list[MBPPQuestion],
    seed: int,
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
) -> DatasetSplit:
    """Cut the fixed split. The same questions and seed always give the same cut.

    Sorting by task id before shuffling makes the result independent of the
    input order, so a different file order cannot change an arm's data.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetError(f"seed must be an integer, got {seed!r}")
    if train_size < 0 or test_size < 0:
        raise DatasetError("split sizes must not be negative")

    expected = train_size + test_size
    if len(questions) != expected:
        raise DatasetError(
            f"expected {expected} questions for a {train_size}/{test_size} split, "
            f"got {len(questions)}"
        )

    ids = [question.task_id for question in questions]
    if len(set(ids)) != len(ids):
        raise DatasetError("questions contain duplicate task ids")

    ordered = sorted(questions, key=lambda question: question.task_id)
    random.Random(seed).shuffle(ordered)
    return DatasetSplit(
        train=tuple(ordered[:train_size]),
        test=tuple(ordered[train_size:expected]),
    )


def load_split(
    path: str | Path = DEFAULT_DATA_PATH,
    *,
    seed: int,
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
) -> DatasetSplit:
    """Load the merged dataset and cut it. The entry point every arm uses."""
    return split_questions(load_dataset(path), seed, train_size, test_size)


def prompt_id_for(question: MBPPQuestion) -> str:
    """The stable prompt id for one question."""
    return f"mbpp-{question.task_id}"


def format_prompt(question: MBPPQuestion) -> str:
    """The exact text sent to the model. One format for every arm."""
    return PROMPT_TEMPLATE.format(text=question.text)


def prompt_for_question(question: MBPPQuestion) -> Prompt:
    """The Experiment prompt for one question."""
    return Prompt(prompt_id=prompt_id_for(question), problem=format_prompt(question))


def prompts_for_questions(questions: Iterable[MBPPQuestion]) -> list[Prompt]:
    """The single path from questions to prompts. Arms must use this function."""
    return [prompt_for_question(question) for question in questions]
