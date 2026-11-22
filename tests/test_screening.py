"""Tests for the difficulty screen command and its writer.

The screen itself is pure orchestration over the generator and test runner
seams, so the fake path covers the whole writer, summary, and resume behaviour
without a GPU. The GPU path is reviewed only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from rlvr.cli import build_generator, main
from rlvr.config import load_config
from rlvr.experiment import Generation, Outcome, Prompt
from rlvr.fake import FakeGenerator, FakeTestRunner
from rlvr.screening import (
    DEFAULT_SCREEN_SAMPLES,
    HISTOGRAM_LABELS,
    ROWS_NAME,
    SCREEN_CHUNK_PROMPTS,
    SUMMARY_NAME,
    ScreenError,
    ScreenRow,
    ScreenStore,
    screen,
    summarize,
)

QUESTIONS = [
    Prompt("mbpp-0001", "problem one"),
    Prompt("mbpp-0002", "problem two"),
    Prompt("mbpp-0003", "problem three"),
]


def write_questions(path: Path, count: int = 3) -> Path:
    rows = [
        {
            "task_id": index + 1,
            "text": f"Return the number {index}.",
            "test_list": [f"assert solution() == {index}"],
            "test_setup_code": "",
            "source_split": "train",
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


class CountingGenerator:
    """Returns one fixed code per prompt and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float, bool]] = []

    def generate(
        self,
        prompt: Prompt,
        *,
        n: int,
        temperature: float,
        thinking: bool = True,
    ) -> list[Generation]:
        self.calls.append((prompt.prompt_id, n, temperature, thinking))
        generation = Generation(
            reasoning="",
            code="def solution():\n    return 0\n",
            prompt_tokens=1,
            completion_tokens=1,
        )
        return [generation] * n


def read_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_summary_puts_rates_in_tenth_buckets_and_counts_the_bands() -> None:
    rows = [
        ScreenRow("a", 10, 0, 0.0),
        ScreenRow("b", 10, 3, 0.3),
        ScreenRow("c", 10, 7, 0.7),
        ScreenRow("d", 10, 9, 0.9),
        ScreenRow("e", 10, 10, 1.0),
    ]

    summary = summarize(rows, samples_per_question=10)

    assert summary["questions"] == 5
    assert summary["samples_per_question"] == 10
    assert summary["zero_pass"] == 1
    assert summary["partial_pass"] == 3
    assert summary["full_pass"] == 1
    assert summary["pass_rate_histogram"] == {
        "0.0": 1,
        "0.1": 0,
        "0.2": 0,
        "0.3": 1,
        "0.4": 0,
        "0.5": 0,
        "0.6": 0,
        "0.7": 1,
        "0.8": 0,
        "0.9": 1,
        "1.0": 1,
    }
    assert sum(summary["pass_rate_histogram"].values()) == 5
    assert list(summary["pass_rate_histogram"]) == list(HISTOGRAM_LABELS)


def test_summary_at_sixteen_samples_counts_zero_partial_and_full() -> None:
    rows = [
        ScreenRow("a", 16, 0, 0.0),
        ScreenRow("b", 16, 1, 0.0625),
        ScreenRow("c", 16, 8, 0.5),
        ScreenRow("d", 16, 15, 0.9375),
        ScreenRow("e", 16, 16, 1.0),
    ]

    summary = summarize(rows, samples_per_question=DEFAULT_SCREEN_SAMPLES)

    assert summary["questions"] == 5
    assert summary["samples_per_question"] == 16
    assert summary["zero_pass"] == 1
    assert summary["partial_pass"] == 3
    assert summary["full_pass"] == 1
    assert summary["pass_rate_histogram"] == {
        "0.0": 2,
        "0.1": 0,
        "0.2": 0,
        "0.3": 0,
        "0.4": 0,
        "0.5": 1,
        "0.6": 0,
        "0.7": 0,
        "0.8": 0,
        "0.9": 1,
        "1.0": 1,
    }
    assert sum(summary["pass_rate_histogram"].values()) == 5


def test_summary_rejects_a_bad_sample_count() -> None:
    with pytest.raises(ScreenError):
        summarize([], samples_per_question=0)
    with pytest.raises(ScreenError):
        summarize([], samples_per_question=True)


def test_summary_rejects_rows_with_a_different_sample_count() -> None:
    with pytest.raises(ScreenError, match="not 4"):
        summarize(
            [ScreenRow("mbpp-0001", 2, 1, 0.5)],
            samples_per_question=4,
        )


def test_store_reads_nothing_before_the_first_row(tmp_path: Path) -> None:
    assert ScreenStore(tmp_path / "out").read_rows() == []


def test_store_reads_back_the_rows_it_wrote(tmp_path: Path) -> None:
    store = ScreenStore(tmp_path / "out")
    store.append_row(ScreenRow("mbpp-0001", 4, 1, 0.25))
    store.append_row(ScreenRow("mbpp-0002", 4, 4, 1.0))

    assert store.read_rows() == [
        ScreenRow("mbpp-0001", 4, 1, 0.25),
        ScreenRow("mbpp-0002", 4, 4, 1.0),
    ]


def test_store_drops_and_removes_a_torn_last_line(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    good = json.dumps(
        {"prompt_id": "mbpp-0001", "samples": 2, "passed": 1, "pass_rate": 0.5}
    )
    (directory / ROWS_NAME).write_text(
        good + "\n" + '{"prompt_id": "mbpp-0002", "sampl', encoding="utf-8"
    )

    rows = ScreenStore(directory).read_rows()

    assert rows == [ScreenRow("mbpp-0001", 2, 1, 0.5)]
    repaired = (directory / ROWS_NAME).read_text(encoding="utf-8")
    assert repaired == good + "\n"
    assert ScreenStore(directory).read_rows() == rows


def test_store_rewrites_a_complete_final_row_without_a_newline(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    first = json.dumps(
        {"prompt_id": "mbpp-0001", "samples": 2, "passed": 1, "pass_rate": 0.5}
    )
    second = json.dumps(
        {"prompt_id": "mbpp-0002", "samples": 2, "passed": 2, "pass_rate": 1.0}
    )
    path = directory / ROWS_NAME
    path.write_text(first + "\n" + second, encoding="utf-8")

    rows = ScreenStore(directory).read_rows()

    assert rows == [
        ScreenRow("mbpp-0001", 2, 1, 0.5),
        ScreenRow("mbpp-0002", 2, 2, 1.0),
    ]
    assert path.read_text(encoding="utf-8") == first + "\n" + second + "\n"
    assert ScreenStore(directory).read_rows() == rows


def test_store_rejects_a_bad_line_that_was_terminated(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    (directory / ROWS_NAME).write_text(
        '{"prompt_id": "mbpp-0001", "samples": 2, "passed": 1, "pass_rate": 0.5}\n'
        "not json at all\n",
        encoding="utf-8",
    )

    with pytest.raises(ScreenError, match="line 2"):
        ScreenStore(directory).read_rows()


def test_store_rejects_a_duplicate_question_id(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    line = json.dumps(
        {"prompt_id": "mbpp-0001", "samples": 2, "passed": 1, "pass_rate": 0.5}
    )
    (directory / ROWS_NAME).write_text(line + "\n" + line + "\n", encoding="utf-8")

    with pytest.raises(ScreenError, match="duplicate"):
        ScreenStore(directory).read_rows()


def test_store_rejects_a_rate_that_does_not_match_the_counts(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    (directory / ROWS_NAME).write_text(
        '{"prompt_id": "mbpp-0001", "samples": 4, "passed": 1, "pass_rate": 0.9}\n',
        encoding="utf-8",
    )

    with pytest.raises(ScreenError, match="pass_rate"):
        ScreenStore(directory).read_rows()


def test_screen_writes_one_row_per_question_and_a_summary(tmp_path: Path) -> None:
    result = screen(
        QUESTIONS,
        samples=3,
        directory=tmp_path / "out",
        generator=FakeGenerator(),
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )

    assert result.resumed == 0
    assert [row.prompt_id for row in result.rows] == [p.prompt_id for p in QUESTIONS]
    rows = read_lines(result.directory / ROWS_NAME)
    assert [row["prompt_id"] for row in rows] == [p.prompt_id for p in QUESTIONS]
    assert all(row["samples"] == 3 for row in rows)
    summary = json.loads((result.directory / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary == dict(result.summary)
    assert summary["questions"] == 3
    assert summary["zero_pass"] + summary["partial_pass"] + summary["full_pass"] == 3
    assert sum(summary["pass_rate_histogram"].values()) == 3
    assert sorted(p.name for p in result.directory.iterdir()) == [
        ROWS_NAME,
        SUMMARY_NAME,
    ]


def test_screen_samples_with_thinking_off_at_the_requested_count(tmp_path: Path) -> None:
    generator = CountingGenerator()

    screen(
        QUESTIONS,
        samples=4,
        directory=tmp_path / "out",
        generator=generator,
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )

    assert generator.calls == [
        ("mbpp-0001", 4, 0.9, False),
        ("mbpp-0002", 4, 0.9, False),
        ("mbpp-0003", 4, 0.9, False),
    ]


def test_the_screen_defaults_to_sixteen_samples(tmp_path: Path) -> None:
    generator = CountingGenerator()

    result = screen(
        QUESTIONS,
        directory=tmp_path / "out",
        generator=generator,
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )

    assert DEFAULT_SCREEN_SAMPLES == 16
    assert generator.calls == [
        ("mbpp-0001", 16, 0.9, False),
        ("mbpp-0002", 16, 0.9, False),
        ("mbpp-0003", 16, 0.9, False),
    ]
    assert [row.samples for row in result.rows] == [16, 16, 16]
    assert result.summary["samples_per_question"] == 16


def test_written_rows_keep_the_four_pool_fields(tmp_path: Path) -> None:
    from rlvr.pools import from_rows

    result = screen(
        QUESTIONS,
        directory=tmp_path / "out",
        generator=FakeGenerator(),
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )

    rows = read_lines(result.directory / ROWS_NAME)
    assert len(rows) == 3
    for row in rows:
        assert list(row) == ["prompt_id", "samples", "passed", "pass_rate"]
        assert isinstance(row["prompt_id"], str) and row["prompt_id"]
        assert row["samples"] == 16
        assert isinstance(row["passed"], int)
        assert 0 <= row["passed"] <= 16
        assert row["pass_rate"] == row["passed"] / row["samples"]

    pools = from_rows(rows, samples=16)
    assert pools.samples == 16
    assert len(pools.train) + len(pools.retired) + len(pools.parked) == 3


def test_screen_resumes_from_the_rows_already_written(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    first = {
        "prompt_id": "mbpp-0001",
        "samples": 2,
        "passed": 1,
        "pass_rate": 0.5,
    }
    (directory / ROWS_NAME).write_text(json.dumps(first) + "\n", encoding="utf-8")
    generator = CountingGenerator()

    result = screen(
        QUESTIONS,
        samples=2,
        directory=directory,
        generator=generator,
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )

    assert result.resumed == 1
    assert [call[0] for call in generator.calls] == ["mbpp-0002", "mbpp-0003"]
    assert result.rows[0] == ScreenRow("mbpp-0001", 2, 1, 0.5)
    summary = json.loads((directory / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["partial_pass"] == 1
    assert summary["questions"] == 3


def test_screen_resumes_after_a_torn_last_row(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    first = {
        "prompt_id": "mbpp-0001",
        "samples": 2,
        "passed": 2,
        "pass_rate": 1.0,
    }
    (directory / ROWS_NAME).write_text(
        json.dumps(first) + "\n" + '{"prompt_id": "mbpp-0002", "samp',
        encoding="utf-8",
    )

    result = screen(
        QUESTIONS,
        samples=2,
        directory=directory,
        generator=FakeGenerator(),
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )

    assert result.resumed == 1
    assert [row.prompt_id for row in result.rows] == [p.prompt_id for p in QUESTIONS]
    rows = read_lines(directory / ROWS_NAME)
    assert [row["prompt_id"] for row in rows] == [p.prompt_id for p in QUESTIONS]
    assert (directory / ROWS_NAME).read_text(encoding="utf-8").endswith("\n")


def test_screen_rejects_a_resume_with_a_different_sample_count(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    (directory / ROWS_NAME).write_text(
        json.dumps(
            {"prompt_id": "mbpp-0001", "samples": 3, "passed": 1, "pass_rate": 1 / 3}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ScreenError, match="3 samples, not 2"):
        screen(
            QUESTIONS,
            samples=2,
            directory=directory,
            generator=FakeGenerator(),
            test_runner=FakeTestRunner(),
            temperature=0.9,
        )


def test_screen_rejects_rows_from_another_question_set(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    directory.mkdir()
    (directory / ROWS_NAME).write_text(
        json.dumps(
            {"prompt_id": "mbpp-9999", "samples": 2, "passed": 1, "pass_rate": 0.5}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ScreenError, match="not in this question set"):
        screen(
            QUESTIONS,
            samples=2,
            directory=directory,
            generator=FakeGenerator(),
            test_runner=FakeTestRunner(),
            temperature=0.9,
        )


def test_screen_rejects_an_empty_question_set(tmp_path: Path) -> None:
    with pytest.raises(ScreenError, match="at least one question"):
        screen(
            [],
            samples=2,
            directory=tmp_path / "out",
            generator=FakeGenerator(),
            test_runner=FakeTestRunner(),
            temperature=0.9,
        )


def test_screen_rejects_a_generator_that_returns_the_wrong_count(
    tmp_path: Path,
) -> None:
    class ShortGenerator(CountingGenerator):
        def generate(self, prompt, *, n, temperature, thinking=True):
            return super().generate(prompt, n=n, temperature=temperature)[:1]

    with pytest.raises(ScreenError, match="expected 3"):
        screen(
            QUESTIONS,
            samples=3,
            directory=tmp_path / "out",
            generator=ShortGenerator(),
            test_runner=FakeTestRunner(),
            temperature=0.9,
        )


class NumberedGenerator:
    """Returns one distinct code per sample, numbered from zero."""

    def generate(
        self,
        prompt: Prompt,
        *,
        n: int,
        temperature: float,
        thinking: bool = True,
    ) -> list[Generation]:
        return [
            Generation(
                reasoning="",
                code=f"def solution():\n    return {index}\n",
                prompt_tokens=1,
                completion_tokens=1,
            )
            for index in range(n)
        ]


def numbered_outcome(code: str) -> Outcome:
    """A deterministic outcome: even sample numbers pass, odd ones fail."""
    value = int(code.rsplit("return ", 1)[1])
    passed = 1 if value % 2 == 0 else 0
    return Outcome(
        tests_passed=passed,
        tests_total=1,
        error_class="pass" if passed else "wrong_answer",
    )


class SerialRecordingRunner:
    """Scores one candidate at a time and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(self, prompt: Prompt, code: str) -> Outcome:
        self.calls.append((prompt.prompt_id, code))
        return numbered_outcome(code)


class BatchRecordingRunner:
    """Scores whole questions in one call and records every batch."""

    def __init__(self) -> None:
        self.batches: list[list[tuple[str, str]]] = []
        self.single_calls = 0

    def run(self, prompt: Prompt, code: str) -> Outcome:
        self.single_calls += 1
        return numbered_outcome(code)

    def run_batch(self, items: Sequence[tuple[Prompt, str]]) -> list[Outcome]:
        self.batches.append([(prompt.prompt_id, code) for prompt, code in items])
        return [numbered_outcome(code) for _, code in items]


def test_screen_scores_each_question_in_one_batch(tmp_path: Path) -> None:
    runner = BatchRecordingRunner()

    result = screen(
        QUESTIONS,
        samples=4,
        directory=tmp_path / "out",
        generator=NumberedGenerator(),
        test_runner=runner,
        temperature=0.9,
    )

    assert [len(batch) for batch in runner.batches] == [4, 4, 4]
    assert [batch[0][0] for batch in runner.batches] == [
        prompt.prompt_id for prompt in QUESTIONS
    ]
    assert all(
        {prompt_id for prompt_id, _ in batch} == {batch[0][0]}
        for batch in runner.batches
    )
    assert runner.single_calls == 0
    assert [row.passed for row in result.rows] == [2, 2, 2]


def test_a_batched_screen_matches_the_serial_rows(tmp_path: Path) -> None:
    serial = screen(
        QUESTIONS,
        samples=4,
        directory=tmp_path / "serial",
        generator=NumberedGenerator(),
        test_runner=SerialRecordingRunner(),
        temperature=0.9,
    )
    batched = screen(
        QUESTIONS,
        samples=4,
        directory=tmp_path / "batched",
        generator=NumberedGenerator(),
        test_runner=BatchRecordingRunner(),
        temperature=0.9,
    )

    assert batched.rows == serial.rows
    assert dict(batched.summary) == dict(serial.summary)
    assert (batched.directory / ROWS_NAME).read_bytes() == (
        serial.directory / ROWS_NAME
    ).read_bytes()
    assert (batched.directory / SUMMARY_NAME).read_bytes() == (
        serial.directory / SUMMARY_NAME
    ).read_bytes()


class ChunkRecordingGenerator:
    """Returns one group per prompt and records every batch call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, float, bool]] = []

    def generate_batch(
        self,
        prompts: Sequence[Prompt],
        *,
        n: int,
        temperature: float,
        thinking: bool = True,
    ) -> list[list[Generation]]:
        prompt_list = list(prompts)
        self.calls.append(
            (
                [prompt.prompt_id for prompt in prompt_list],
                n,
                temperature,
                thinking,
            )
        )
        return [
            [
                Generation(
                    reasoning="",
                    code="def solution():\n    return 0\n",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            ]
            * n
            for _ in prompt_list
        ]


class AlwaysPassRunner:
    """Scores every candidate as passing without running anything."""

    def run(self, prompt: Prompt, code: str) -> Outcome:
        return Outcome(tests_passed=1, tests_total=1, error_class="pass")

    def run_batch(self, items: Sequence[tuple[Prompt, str]]) -> list[Outcome]:
        return [
            Outcome(tests_passed=1, tests_total=1, error_class="pass")
            for _ in items
        ]


def test_screen_generates_a_whole_chunk_in_one_call(tmp_path: Path) -> None:
    generator = ChunkRecordingGenerator()

    result = screen(
        QUESTIONS,
        samples=4,
        directory=tmp_path / "out",
        generator=generator,
        test_runner=AlwaysPassRunner(),
        temperature=0.9,
    )

    assert SCREEN_CHUNK_PROMPTS == 32
    assert generator.calls == [
        ([prompt.prompt_id for prompt in QUESTIONS], 4, 0.9, False)
    ]
    assert [row.passed for row in result.rows] == [4, 4, 4]


def test_screen_splits_711_questions_into_23_generate_calls(
    tmp_path: Path,
) -> None:
    prompts = [
        Prompt(f"mbpp-{index:04d}", f"problem {index}") for index in range(711)
    ]
    generator = ChunkRecordingGenerator()

    screen(
        prompts,
        directory=tmp_path / "out",
        generator=generator,
        test_runner=AlwaysPassRunner(),
        temperature=0.9,
    )

    assert len(generator.calls) == 23
    assert [len(call[0]) for call in generator.calls] == [32] * 22 + [7]
    assert [
        prompt_id
        for call in generator.calls
        for prompt_id in call[0]
    ] == [prompt.prompt_id for prompt in prompts]
    assert all(call[1] == DEFAULT_SCREEN_SAMPLES for call in generator.calls)


class OffsetNumberedGenerator:
    """Distinct codes per prompt and sample. No batch method, so it falls back."""

    def generate(
        self,
        prompt: Prompt,
        *,
        n: int,
        temperature: float,
        thinking: bool = True,
    ) -> list[Generation]:
        offset = int(prompt.prompt_id.rsplit("-", 1)[1])
        return [
            Generation(
                reasoning="",
                code=f"def solution():\n    return {(offset + index) % 5}\n",
                prompt_tokens=1,
                completion_tokens=1,
            )
            for index in range(n)
        ]


class BatchedOffsetNumberedGenerator(OffsetNumberedGenerator):
    """The same codes, produced through one call per chunk."""

    def __init__(self) -> None:
        self.batch_calls = 0

    def generate_batch(
        self,
        prompts: Sequence[Prompt],
        *,
        n: int,
        temperature: float,
        thinking: bool = True,
    ) -> list[list[Generation]]:
        self.batch_calls += 1
        return [
            self.generate(prompt, n=n, temperature=temperature, thinking=thinking)
            for prompt in prompts
        ]


def test_chunked_generation_writes_the_same_rows_as_the_per_prompt_path(
    tmp_path: Path,
) -> None:
    prompts = [
        Prompt(f"mbpp-{index:04d}", f"problem {index}") for index in range(9)
    ]
    serial = screen(
        prompts,
        samples=4,
        directory=tmp_path / "serial",
        generator=OffsetNumberedGenerator(),
        test_runner=SerialRecordingRunner(),
        temperature=0.9,
    )
    generator = BatchedOffsetNumberedGenerator()
    batched = screen(
        prompts,
        samples=4,
        directory=tmp_path / "batched",
        generator=generator,
        test_runner=SerialRecordingRunner(),
        temperature=0.9,
    )

    assert generator.batch_calls == 1
    assert batched.rows == serial.rows
    assert dict(batched.summary) == dict(serial.summary)
    assert (batched.directory / ROWS_NAME).read_bytes() == (
        serial.directory / ROWS_NAME
    ).read_bytes()
    assert (batched.directory / SUMMARY_NAME).read_bytes() == (
        serial.directory / SUMMARY_NAME
    ).read_bytes()


def test_two_fake_screens_produce_the_same_bytes(tmp_path: Path) -> None:
    first = screen(
        QUESTIONS,
        samples=4,
        directory=tmp_path / "first",
        generator=FakeGenerator(),
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )
    second = screen(
        QUESTIONS,
        samples=4,
        directory=tmp_path / "second",
        generator=FakeGenerator(),
        test_runner=FakeTestRunner(),
        temperature=0.9,
    )

    assert (first.directory / ROWS_NAME).read_bytes() == (
        second.directory / ROWS_NAME
    ).read_bytes()
    assert (first.directory / SUMMARY_NAME).read_bytes() == (
        second.directory / SUMMARY_NAME
    ).read_bytes()


def test_cli_screen_fake_writes_the_table_and_needs_no_gpu(
    tmp_path: Path, capsys
) -> None:
    questions = write_questions(tmp_path / "questions.jsonl")
    output = tmp_path / "screen"

    code = main(
        [
            "screen",
            "--questions",
            str(questions),
            "--samples",
            "2",
            "--output",
            str(output),
            "--fake",
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "screen: questions=3 samples=2 resumed=0" in printed
    assert f"rows: {output / ROWS_NAME}" in printed
    assert (output / ROWS_NAME).is_file()
    assert (output / SUMMARY_NAME).is_file()
    assert not (output / "checkpoints").exists()
    assert not (output / "meta.json").exists()


def test_cli_screen_uses_sixteen_samples_when_the_flag_is_absent(
    tmp_path: Path, capsys
) -> None:
    questions = write_questions(tmp_path / "questions.jsonl")
    output = tmp_path / "screen"

    code = main(
        [
            "screen",
            "--questions",
            str(questions),
            "--output",
            str(output),
            "--fake",
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "samples=16" in printed
    rows = read_lines(output / ROWS_NAME)
    assert [row["samples"] for row in rows] == [16, 16, 16]
    summary = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["samples_per_question"] == 16
    assert (
        summary["zero_pass"] + summary["partial_pass"] + summary["full_pass"] == 3
    )
    assert sum(summary["pass_rate_histogram"].values()) == 3


def test_cli_screen_rerun_skips_done_questions(tmp_path: Path, capsys) -> None:
    questions = write_questions(tmp_path / "questions.jsonl")
    output = tmp_path / "screen"
    args = [
        "screen",
        "--questions",
        str(questions),
        "--samples",
        "2",
        "--output",
        str(output),
        "--fake",
    ]
    assert main(args) == 0
    capsys.readouterr()
    before = (output / ROWS_NAME).read_bytes()

    assert main(args) == 0
    printed = capsys.readouterr().out

    assert "resumed=3" in printed
    assert (output / ROWS_NAME).read_bytes() == before
    assert len(read_lines(output / ROWS_NAME)) == 3


def test_cli_screen_rerun_repairs_a_torn_last_row(tmp_path: Path, capsys) -> None:
    questions = write_questions(tmp_path / "questions.jsonl")
    output = tmp_path / "screen"
    args = [
        "screen",
        "--questions",
        str(questions),
        "--samples",
        "2",
        "--output",
        str(output),
        "--fake",
    ]
    assert main(args) == 0
    capsys.readouterr()

    path = output / ROWS_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2],
        encoding="utf-8",
    )

    assert main(args) == 0
    printed = capsys.readouterr().out

    assert "resumed=2" in printed
    rows = read_lines(path)
    assert [row["prompt_id"] for row in rows] == [
        "mbpp-1",
        "mbpp-2",
        "mbpp-3",
    ]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_cli_screen_rejects_a_sample_count_below_one(tmp_path: Path, capsys) -> None:
    questions = write_questions(tmp_path / "questions.jsonl")

    code = main(
        [
            "screen",
            "--questions",
            str(questions),
            "--samples",
            "0",
            "--output",
            str(tmp_path / "screen"),
            "--fake",
        ]
    )

    assert code == 2
    assert "--samples" in capsys.readouterr().err
    assert not (tmp_path / "screen").exists()


def test_cli_screen_reports_a_missing_question_file(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "screen",
            "--questions",
            str(tmp_path / "absent.jsonl"),
            "--samples",
            "2",
            "--output",
            str(tmp_path / "screen"),
            "--fake",
        ]
    )

    assert code == 2
    assert "question set" in capsys.readouterr().err


def test_cli_screen_reports_an_empty_question_file(tmp_path: Path, capsys) -> None:
    questions = tmp_path / "empty.jsonl"
    questions.write_text("", encoding="utf-8")

    code = main(
        [
            "screen",
            "--questions",
            str(questions),
            "--samples",
            "2",
            "--output",
            str(tmp_path / "screen"),
            "--fake",
        ]
    )

    assert code == 2
    assert "empty" in capsys.readouterr().err


def test_cli_screen_without_fake_asks_for_the_train_extra(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import rlvr.qwen
    import rlvr.vllm_backend as backend

    def missing(*args, **kwargs):
        raise ImportError("simulated missing torch")

    class StubEngine:
        def __init__(self, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(rlvr.qwen, "QwenGenerator", missing)
    monkeypatch.setattr(backend, "VLLMEngine", StubEngine)
    questions = write_questions(tmp_path / "questions.jsonl")
    output = tmp_path / "screen"

    code = main(
        [
            "screen",
            "--questions",
            str(questions),
            "--samples",
            "2",
            "--output",
            str(output),
        ]
    )

    assert code == 2
    message = capsys.readouterr().err
    assert "pip install -e '.[vllm]'" in message
    assert "--fake" in message
    assert not output.exists()


def test_the_screen_asks_for_a_non_trainable_base_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from rlvr import cli

    captured: dict[str, object] = {}

    def fake_build(config, *, fake, trainable=True):
        captured["trainable"] = trainable
        return FakeGenerator()

    monkeypatch.setattr(cli, "build_generator", fake_build)
    questions = write_questions(tmp_path / "questions.jsonl")

    assert main(
        [
            "screen",
            "--questions",
            str(questions),
            "--samples",
            "2",
            "--output",
            str(tmp_path / "screen"),
            "--fake",
        ]
    ) == 0

    assert captured == {"trainable": False}


def test_build_generator_wraps_every_real_generator_in_the_vllm_backend(
    write_config, monkeypatch
) -> None:
    import rlvr.qwen
    import rlvr.vllm_backend as backend

    engines: list[dict[str, object]] = []
    trainers: list[object] = []

    class StubTrainer:
        def __init__(self, **kwargs) -> None:
            self.trainable = bool(kwargs["trainable"])
            self.max_completion_tokens = kwargs["max_completion_tokens"]

    class StubEngine:
        def __init__(self, **kwargs) -> None:
            engines.append(kwargs)

        def close(self) -> None:
            pass

    class StubGenerator:
        def __init__(self, *, trainer, engine) -> None:
            trainers.append(trainer)

    monkeypatch.setattr(rlvr.qwen, "QwenGenerator", StubTrainer)
    monkeypatch.setattr(backend, "VLLMEngine", StubEngine)
    monkeypatch.setattr(backend, "VLLMGenerator", StubGenerator)
    config = load_config(write_config())

    build_generator(config, fake=False, trainable=False)
    build_generator(config, fake=False)
    build_generator(config, fake=True)

    assert engines[0]["max_model_len"] == (
        config.max_prompt_tokens + config.max_completion_tokens
    )
    assert trainers[0].trainable is False
    assert trainers[1].trainable is True
    assert trainers[0].max_completion_tokens == config.max_completion_tokens
    assert len(engines) == 2
    assert len(trainers) == 2


def test_build_screen_test_runner_uses_fakes_or_docker() -> None:
    from rlvr.cli import build_screen_test_runner
    from rlvr.data import load_dataset
    from rlvr.docker_runner import DockerTestRunner

    questions = load_dataset()
    assert isinstance(build_screen_test_runner(questions, fake=True), FakeTestRunner)
    # Construction reads the question rows only. No container is started.
    assert isinstance(build_screen_test_runner(questions, fake=False), DockerTestRunner)
