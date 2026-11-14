"""Tests for the pure stage builder and the stage loop in Experiment.

No model, Docker, or GPU is involved. The builder is tested directly, and the
loop is driven through the Experiment seam with small stubs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.batching import BatchSizePolicy
from rlvr.config import load_config
from rlvr.experiment import Experiment, Generation, Outcome, Prompt, Stage
from rlvr.stages import (
    BANDS,
    DEFAULT_CARRY_OVER_RATIO,
    LabelledRow,
    StageError,
    build_stages,
    load_band_labels,
)

CONFIG_TEXT = """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "r"

[judge]
model = "fake-judge-model"

[data]
split_seed = 1

[fake]
steps = 5
"""


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


class StubGenerator:
    def __init__(self) -> None:
        self.update_calls: list[str] = []

    def generate(self, prompt: Prompt, *, n: int, temperature: float) -> list[Generation]:
        return [
            Generation(
                reasoning="trace",
                code=f"# {prompt.prompt_id}",
                prompt_tokens=5,
                completion_tokens=5,
            )
            for _ in range(n)
        ]

    def update(self, prompt, generations, rewards, *, measure_entropy: bool = False):
        self.update_calls.append(prompt.prompt_id)
        return None

    def save_checkpoint(self, directory) -> None:
        pass

    def load_checkpoint(self, directory) -> None:
        pass


class StubRunner:
    __test__ = False

    def __init__(self, passing: set[str]) -> None:
        self.passing = passing
        self.calls: list[str] = []

    def run(self, prompt: Prompt, code: str) -> Outcome:
        self.calls.append(prompt.prompt_id)
        if prompt.prompt_id in self.passing:
            return Outcome(tests_passed=2, tests_total=2, error_class="pass")
        return Outcome(tests_passed=1, tests_total=2, error_class="wrong_answer")


class StubJudge:
    def score_output(self, prompt, code):
        raise AssertionError("the code arm must not judge")

    score_process = score_output
    score_outputs = score_output
    score_processes = score_output


class RecordingStore:
    def __init__(self) -> None:
        self.rollouts = []
        self.metrics = []
        self.meta = None
        self.eval = None
        self.timestamp = "20261112T000000Z"
        self.run_dir = Path("runs/test/20261112T000000Z")
        self.checkpoints_dir = self.run_dir / "checkpoints"

    def write_meta(self, meta) -> None:
        self.meta = meta

    def append_rollout(self, record) -> None:
        self.rollouts.append(record)

    def append_metric(self, row) -> None:
        self.metrics.append(row)

    def write_eval(self, payload) -> None:
        self.eval = payload


def prompt(name: str) -> Prompt:
    return Prompt(prompt_id=name, problem=f"problem {name}")


def prompts_by_step(store: RecordingStore) -> list[str]:
    """One prompt id per step, taken from that step's first rollout."""
    result: list[str] = []
    seen: set[int] = set()
    for record in store.rollouts:
        if record.step not in seen:
            seen.add(record.step)
            result.append(record.prompt_id)
    return result


def run_with_stages(tmp_path: Path, stages, *, passing: set[str], steps: int = 5):
    text = CONFIG_TEXT.replace("steps = 5", f"steps = {steps}")
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    store = RecordingStore()
    result = Experiment(
        arm="code-score",
        config=load_config(config_path),
        generator=StubGenerator(),
        judge=StubJudge(),
        test_runner=StubRunner(passing),
        store=store,
        prompts=[stage.train[0] for stage in stages if stage.train],
        stages=stages,
        eval_prompts=[prompt("eval-0000")],
        # These tests were written for one group per step. Batch sizing inside
        # a stage has its own test in test_batching.py.
        batch_policy=BatchSizePolicy(current=1, max_prompts=1),
    ).run()
    return store, result


def test_a_full_pass_removes_the_question_but_partial_credit_does_not(
    tmp_path: Path,
) -> None:
    stages = (Stage("very easy", (prompt("q0"), prompt("q1"))),)
    store, result = run_with_stages(tmp_path, stages, passing={"q0"}, steps=4)

    assert prompts_by_step(store) == ["q0", "q1", "q1", "q1"]
    assert result.stop_reason == "completed"
    assert [row["stage"] for row in store.metrics] == ["very easy"] * 4


def test_records_and_metrics_carry_the_stage_name(tmp_path: Path) -> None:
    stages = (Stage("very easy", (prompt("q0"),)),)
    store, _ = run_with_stages(tmp_path, stages, passing=set(), steps=2)

    assert {record.stage for record in store.rollouts} == {"very easy"}
    assert [row["stage"] for row in store.metrics] == ["very easy", "very easy"]


def test_an_exhausted_pool_advances_to_the_next_stage(tmp_path: Path) -> None:
    stages = (
        Stage("very easy", (prompt("q0"), prompt("q1"))),
        Stage("medium", (prompt("q2"),)),
    )
    store, result = run_with_stages(tmp_path, stages, passing={"q0", "q1", "q2"}, steps=5)

    assert prompts_by_step(store) == ["q0", "q1", "q2"]
    assert [row["stage"] for row in store.metrics] == ["very easy", "very easy", "medium"]
    assert result.steps == 3
    assert result.stop_reason == "curriculum_complete"
    assert result.stage == "medium"


def test_exhausting_the_last_stage_ends_the_run_with_a_clear_reason(
    tmp_path: Path,
) -> None:
    stages = (Stage("very easy", (prompt("q0"),)),)
    store, result = run_with_stages(tmp_path, stages, passing={"q0"}, steps=5)

    assert result.steps == 1
    assert result.stop_reason == "curriculum_complete"
    assert store.eval["stop_reason"] == "curriculum_complete"


def test_meta_records_the_stage_plan(tmp_path: Path) -> None:
    stages = (Stage("very easy", (prompt("q0"),), (prompt("held"),)),)
    store, _ = run_with_stages(tmp_path, stages, passing=set(), steps=1)

    curriculum = store.meta["curriculum"]
    assert curriculum["bands"] == list(BANDS)
    assert curriculum["start_stage"] == "very easy"
    assert curriculum["carry_over_ratio"] == DEFAULT_CARRY_OVER_RATIO
    assert curriculum["stages"] == [
        {"name": "very easy", "train": 1, "held_out": 1}
    ]


def test_a_stage_with_no_training_questions_is_skipped(tmp_path: Path) -> None:
    stages = (
        Stage("very easy", ()),
        Stage("medium", (prompt("q0"),)),
    )
    store, result = run_with_stages(tmp_path, stages, passing={"q0"}, steps=3)

    assert prompts_by_step(store) == ["q0"]
    assert [row["stage"] for row in store.metrics] == ["medium"]
    assert result.stop_reason == "curriculum_complete"


def test_stages_with_no_training_questions_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONFIG_TEXT, encoding="utf-8")
    experiment = Experiment(
        arm="code-score",
        config=load_config(config_path),
        generator=StubGenerator(),
        judge=StubJudge(),
        test_runner=StubRunner(set()),
        store=RecordingStore(),
        prompts=[prompt("q0")],
        stages=(Stage("very easy", ()),),
        eval_prompts=[prompt("eval-0000")],
    )
    with pytest.raises(ValueError, match="no training questions"):
        experiment.run()


def test_the_stage_builder_output_feeds_the_loop(tmp_path: Path) -> None:
    rows = [
        LabelledRow("easy-00", "very easy", prompt("easy-00")),
        LabelledRow("medium-00", "medium", prompt("medium-00")),
    ]
    stages = build_stages(rows, seed=4)
    store, result = run_with_stages(
        tmp_path, stages, passing={"easy-00", "medium-00"}, steps=3
    )

    assert [row["stage"] for row in store.metrics] == ["very easy", "medium"]
    assert result.stop_reason == "curriculum_complete"


def test_the_stage_field_round_trips_through_jsonl(tmp_path: Path) -> None:
    from rlvr.schema import RolloutRecord

    stages = (Stage("hard", (prompt("q0"),)),)
    store, _ = run_with_stages(tmp_path, stages, passing=set(), steps=1)
    line = store.rollouts[0].to_json()
    assert json.loads(line)["stage"] == "hard"
    assert RolloutRecord.from_json(line).stage == "hard"
