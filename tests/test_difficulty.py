"""Tests for the difficulty labels and the synthetic very easy set.

Both are committed data files. These tests check the files against the split
and the row contract without a model, a network call, or Docker.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from rlvr.data import (
    DEFAULT_LABELS_PATH,
    TRAIN_SIZE,
    DatasetError,
    DatasetSplit,
    MBPPQuestion,
    load_dataset,
    load_difficulty_labels,
    load_split,
    prompt_for_question,
    prompt_id_for,
)
from rlvr.stages import BANDS
from rlvr.synthetic import (
    DEFAULT_SYNTHETIC_PATH,
    MAX_REFERENCE_LINES,
    MIN_TESTS,
    SYNTHETIC_TASK_ID_START,
    SyntheticError,
    load_synthetic_questions,
)

SPLIT_SEED = 1234


def training_split() -> DatasetSplit:
    return load_split(seed=SPLIT_SEED)


def tiny_split(*ids: int) -> DatasetSplit:
    return DatasetSplit(
        train=tuple(
            MBPPQuestion(
                task_id=task_id,
                text=f"problem {task_id}",
                test_list=("assert True",),
                source_split="train",
            )
            for task_id in ids
        ),
        test=(),
    )


def write_label_file(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def label(prompt_id: str, band: str = "medium") -> dict:
    return {"prompt_id": prompt_id, "band": band}


def write_synthetic_file(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "synthetic.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def synthetic_row(**overrides) -> dict:
    row = {
        "task_id": SYNTHETIC_TASK_ID_START,
        "text": "Write a function to add one.",
        "code": "def add_one(value):\n    return value + 1\n",
        "test_list": [
            "assert add_one(1) == 2",
            "assert add_one(0) == 1",
        ],
        "test_setup_code": "",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Difficulty labels


def test_the_committed_labels_cover_every_training_row_in_split_order() -> None:
    split = training_split()
    labels = load_difficulty_labels(split)

    assert DEFAULT_LABELS_PATH.is_file()
    assert len(split.train) == TRAIN_SIZE
    assert list(labels) == [prompt_id_for(question) for question in split.train]
    assert len(labels) == TRAIN_SIZE


def test_the_committed_labels_use_only_the_four_bands() -> None:
    labels = load_difficulty_labels(training_split())
    assert set(labels.values()) <= set(BANDS)
    assert set(labels.values()) == set(BANDS)


def test_the_committed_band_mix_is_plausible() -> None:
    labels = load_difficulty_labels(training_split())
    counts = {band: list(labels.values()).count(band) for band in BANDS}
    total = sum(counts.values())

    for band, count in counts.items():
        assert count > 0, band
    # Very easy is a small prior, medium is the bulk, and the tail is real.
    assert 0.02 <= counts["very easy"] / total <= 0.10
    assert counts["medium"] == max(counts.values())
    assert (counts["hard"] + counts["very hard"]) / total >= 0.10


def test_a_missing_training_id_is_rejected(tmp_path: Path) -> None:
    path = write_label_file(tmp_path, [label("mbpp-1")])
    with pytest.raises(DatasetError, match="missing 2 training id"):
        load_difficulty_labels(tiny_split(1, 2, 3), path)


def test_an_unknown_prompt_id_is_rejected(tmp_path: Path) -> None:
    path = write_label_file(tmp_path, [label("mbpp-1"), label("mbpp-9")])
    with pytest.raises(DatasetError, match="unknown prompt id"):
        load_difficulty_labels(tiny_split(1, 2), path)


def test_a_duplicate_prompt_id_is_rejected(tmp_path: Path) -> None:
    path = write_label_file(
        tmp_path, [label("mbpp-1"), label("mbpp-1"), label("mbpp-2")]
    )
    with pytest.raises(DatasetError, match="duplicate prompt id"):
        load_difficulty_labels(tiny_split(1, 2), path)


def test_an_unknown_band_is_rejected(tmp_path: Path) -> None:
    path = write_label_file(
        tmp_path, [label("mbpp-1", "easy"), label("mbpp-2")]
    )
    with pytest.raises(DatasetError, match="unknown band"):
        load_difficulty_labels(tiny_split(1, 2), path)


def test_labels_out_of_split_order_are_rejected(tmp_path: Path) -> None:
    path = write_label_file(tmp_path, [label("mbpp-2"), label("mbpp-1")])
    with pytest.raises(DatasetError, match="out of split order"):
        load_difficulty_labels(tiny_split(1, 2), path)


def test_a_missing_label_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_difficulty_labels(tiny_split(1), tmp_path / "absent.jsonl")


def test_the_loader_returns_the_split_order(tmp_path: Path) -> None:
    path = write_label_file(
        tmp_path, [label("mbpp-3", "hard"), label("mbpp-1", "very easy")]
    )
    labels = load_difficulty_labels(tiny_split(3, 1), path)
    assert list(labels) == ["mbpp-3", "mbpp-1"]
    assert labels["mbpp-1"] == "very easy"


# ---------------------------------------------------------------------------
# Synthetic questions


def test_the_committed_synthetic_set_has_fifty_questions() -> None:
    questions = load_synthetic_questions()
    assert DEFAULT_SYNTHETIC_PATH.is_file()
    assert len(questions) == 50


def test_synthetic_ids_cannot_collide_with_mbpp_ids() -> None:
    questions = load_synthetic_questions()
    mbpp_ids = {question.task_id for question in load_dataset()}
    synthetic_ids = {question.task_id for question in questions}

    assert synthetic_ids.isdisjoint(mbpp_ids)
    assert min(synthetic_ids) >= SYNTHETIC_TASK_ID_START
    assert all(
        question.prompt_id.startswith("synthetic-") for question in questions
    )


def test_synthetic_prompt_ids_are_distinct_from_every_mbpp_prompt_id() -> None:
    split = training_split()
    mbpp_prompt_ids = {
        prompt_id_for(question)
        for question in list(split.train) + list(split.test)
    }
    synthetic_prompt_ids = {
        question.prompt_id for question in load_synthetic_questions()
    }
    assert synthetic_prompt_ids.isdisjoint(mbpp_prompt_ids)


def test_every_synthetic_question_has_at_least_two_tests() -> None:
    for question in load_synthetic_questions():
        assert len(question.test_list) >= MIN_TESTS, question.task_id
        assert all(test.strip() for test in question.test_list)


def test_every_reference_solution_is_under_twenty_lines() -> None:
    for question in load_synthetic_questions():
        lines = len(question.code.strip().splitlines())
        assert 1 <= lines <= MAX_REFERENCE_LINES, question.task_id
        assert lines < 20, question.task_id


def test_every_reference_solution_uses_only_the_standard_library() -> None:
    for question in load_synthetic_questions():
        tree = ast.parse(question.code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            assert roots, question.task_id
            assert all(
                root in sys.stdlib_module_names for root in roots
            ), f"{question.task_id} imports {roots}"


def test_every_reference_solution_passes_its_own_tests() -> None:
    for question in load_synthetic_questions():
        namespace: dict[str, object] = {"__name__": "candidate"}
        exec(compile(question.code, "candidate.py", "exec"), namespace)
        for index, test in enumerate(question.test_list):
            try:
                exec(compile(test, f"test_{index}.py", "exec"), namespace)
            except Exception as error:
                raise AssertionError(
                    f"task {question.task_id} failed test {index}: {error!r}"
                ) from error


def test_a_synthetic_question_becomes_a_prompt_with_its_own_id() -> None:
    question = load_synthetic_questions()[0]
    backing = question.as_question()

    assert backing.source_split == "synthetic"
    assert backing.prompt_id == question.prompt_id
    prompt = prompt_for_question(backing)
    assert prompt.prompt_id == question.prompt_id
    assert question.text in prompt.problem


def test_the_synthetic_loader_rejects_a_duplicate_id(tmp_path: Path) -> None:
    path = write_synthetic_file(
        tmp_path, [synthetic_row(), synthetic_row()]
    )
    with pytest.raises(SyntheticError, match="duplicate task_id"):
        load_synthetic_questions(path)


def test_the_synthetic_loader_rejects_fewer_than_two_tests(
    tmp_path: Path,
) -> None:
    path = write_synthetic_file(
        tmp_path, [synthetic_row(test_list=["assert add_one(1) == 2"])]
    )
    with pytest.raises(SyntheticError, match="at least"):
        load_synthetic_questions(path)


def test_the_synthetic_loader_rejects_a_long_solution(tmp_path: Path) -> None:
    code = "\n".join(f"value_{index} = {index}" for index in range(20))
    path = write_synthetic_file(tmp_path, [synthetic_row(code=code)])
    with pytest.raises(SyntheticError, match="limit"):
        load_synthetic_questions(path)


def test_the_synthetic_loader_rejects_an_id_inside_the_mbpp_range(
    tmp_path: Path,
) -> None:
    path = write_synthetic_file(tmp_path, [synthetic_row(task_id=974)])
    with pytest.raises(SyntheticError, match="collides"):
        load_synthetic_questions(path)


def test_the_synthetic_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SyntheticError, match="not found"):
        load_synthetic_questions(tmp_path / "absent.jsonl")
