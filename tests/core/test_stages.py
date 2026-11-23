"""Tests for the pure band helpers that `eval-band` still uses.

No model, Docker, or GPU is involved. The training loop no longer builds stages,
so this module covers only the band builder and the label reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlvr.core.stages import (
    BANDS,
    DEFAULT_CARRY_OVER_RATIO,
    LabelledRow,
    StageError,
    build_stages,
    load_band_labels,
)


def rows_for(band: str, count: int, prefix: str | None = None) -> list[LabelledRow[str]]:
    name = prefix or band.replace(" ", "-")
    return [LabelledRow(f"{name}-{index:02d}", band, f"{name}-{index:02d}") for index in range(count)]


def test_bands_are_fixed_in_order() -> None:
    assert BANDS == ("very easy", "medium", "hard", "very hard")
    stages = build_stages(rows_for("very easy", 1))
    assert [stage.name for stage in stages] == list(BANDS)


def test_the_builder_is_pure_and_ignores_input_order() -> None:
    rows = rows_for("very easy", 10)
    first = build_stages(rows, seed=7)
    second = build_stages(list(reversed(rows)), seed=7)
    assert first == second


def test_each_stage_splits_eighty_twenty() -> None:
    stages = build_stages(rows_for("very easy", 10), seed=7)
    stage = stages[0]
    assert len(stage.train) == 8
    assert len(stage.held_out) == 2
    assert {item for item in stage.train}.isdisjoint(stage.held_out)
    assert set(stage.train) | set(stage.held_out) == {
        f"very-easy-{index:02d}" for index in range(10)
    }


def test_the_default_carry_over_takes_the_previous_band() -> None:
    rows = rows_for("very easy", 10, "easy") + rows_for("medium", 10, "medium")
    stages = build_stages(rows, seed=3)
    easy, medium = stages[0], stages[1]

    carried = [item for item in medium.train if item.startswith("easy-")]
    assert len(carried) == round(len(easy.train) * DEFAULT_CARRY_OVER_RATIO)
    assert set(carried) <= set(easy.train)
    assert set(medium.held_out).isdisjoint(carried)
    # The medium stage never takes from a band after it.
    assert all(item.startswith("medium-") or item.startswith("easy-") for item in medium.train)


def test_the_carry_over_ratio_is_configurable() -> None:
    rows = rows_for("very easy", 10, "easy") + rows_for("medium", 10, "medium")
    stages = build_stages(rows, carry_over_ratio=0.5, seed=3)
    carried = [item for item in stages[1].train if item.startswith("easy-")]
    # The previous band has 8 training questions after its own 80/20 split.
    assert len(carried) == 4


def test_a_chosen_start_stage_carries_nothing_from_before_it() -> None:
    rows = rows_for("very easy", 10, "easy") + rows_for("medium", 10, "medium")
    stages = build_stages(rows, start_stage="medium", seed=3)
    assert [stage.name for stage in stages] == ["medium", "hard", "very hard"]
    assert all(item.startswith("medium-") for item in stages[0].train)


def test_an_unknown_band_is_rejected() -> None:
    with pytest.raises(StageError, match="unknown band"):
        build_stages([LabelledRow("q1", "easy", "q1")])


def test_a_duplicate_prompt_id_is_rejected() -> None:
    with pytest.raises(StageError, match="duplicate"):
        build_stages(
            [
                LabelledRow("q1", "very easy", "q1"),
                LabelledRow("q1", "medium", "q1"),
            ]
        )


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5, "half", True])
def test_a_bad_carry_over_ratio_is_rejected(ratio) -> None:
    with pytest.raises(StageError, match="carry over ratio"):
        build_stages([], carry_over_ratio=ratio)


def test_an_unknown_start_stage_is_rejected() -> None:
    with pytest.raises(StageError, match="start stage"):
        build_stages([], start_stage="easy")


def test_a_bad_seed_is_rejected() -> None:
    with pytest.raises(StageError, match="seed"):
        build_stages([], seed=True)


def test_a_missing_label_file_degrades_to_an_empty_mapping(tmp_path: Path) -> None:
    assert load_band_labels(tmp_path / "absent.jsonl") == {}


def test_the_loader_maps_prompt_id_to_band(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"prompt_id": "mbpp-0001", "band": "very easy"}\n'
        '{"prompt_id": "mbpp-0002", "band": "hard"}\n',
        encoding="utf-8",
    )
    assert load_band_labels(path) == {
        "mbpp-0001": "very easy",
        "mbpp-0002": "hard",
    }


@pytest.mark.parametrize(
    "text, message",
    [
        ('{"prompt_id": "mbpp-0001", "band": "easy"}\n', "unknown band"),
        ('{"prompt_id": "", "band": "medium"}\n', "prompt_id"),
        ('{"prompt_id": "mbpp-0001"}\n', "unknown band"),
        ('{"prompt_id": "mbpp-0001", "band": "medium"}\n'
         '{"prompt_id": "mbpp-0001", "band": "hard"}\n', "duplicate"),
        ("not json\n", "not valid JSON"),
    ],
)
def test_a_bad_label_file_fails_loudly(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(StageError, match=message):
        load_band_labels(path)


def test_a_stage_pool_never_repeats_a_question() -> None:
    rows = rows_for("very easy", 20, "ve") + rows_for("medium", 20, "med")
    for stage in build_stages(rows, seed=11):
        assert len(stage.train) == len(set(stage.train)), stage.name
        assert len(stage.held_out) == len(set(stage.held_out)), stage.name
        assert set(stage.train).isdisjoint(stage.held_out), stage.name
