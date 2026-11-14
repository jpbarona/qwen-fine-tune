"""Tests for the difficulty screen command and its writer.

The screen itself is pure orchestration over the generator and test runner
seams, so the fake path covers the whole writer, summary, and resume behaviour
without a GPU. The GPU path is reviewed only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.cli import build_generator, main
from rlvr.config import load_config
from rlvr.experiment import Generation, Prompt
from rlvr.fake import FakeGenerator, FakeTestRunner
from rlvr.screening import (
    HISTOGRAM_LABELS,
    ROWS_NAME,
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

    def missing(*args, **kwargs):
        raise ImportError("simulated missing torch")

    monkeypatch.setattr(rlvr.qwen, "QwenGenerator", missing)
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
    assert "train extra" in capsys.readouterr().err
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


def test_build_generator_passes_the_base_model_flag_and_the_cap(
    write_config, monkeypatch
) -> None:
    import rlvr.qwen

    captured: list[dict[str, object]] = []

    class StubGenerator:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(rlvr.qwen, "QwenGenerator", StubGenerator)
    config = load_config(write_config())

    build_generator(config, fake=False, trainable=False)
    build_generator(config, fake=False)
    build_generator(config, fake=True)

    assert captured[0]["trainable"] is False
    assert captured[1]["trainable"] is True
    assert captured[0]["max_completion_tokens"] == config.max_completion_tokens
    assert len(captured) == 2


def test_build_screen_test_runner_uses_fakes_or_docker() -> None:
    from rlvr.cli import build_screen_test_runner
    from rlvr.data import load_dataset
    from rlvr.docker_runner import DockerTestRunner

    questions = load_dataset()
    assert isinstance(build_screen_test_runner(questions, fake=True), FakeTestRunner)
    # Construction reads the question rows only. No container is started.
    assert isinstance(build_screen_test_runner(questions, fake=False), DockerTestRunner)
