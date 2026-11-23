"""Tests for the saved MLflow view of the study page.

The view is a plain dict, so the builder test needs no store. The encoder test
follows the MLflow UI format: a `deflate;` prefix, then base64 of a zlib
stream. The writer test uses a real temporary SQLite store.
"""

from __future__ import annotations

import base64
import json
import os
import zlib
from pathlib import Path

# Silence a one time hint MLflow prints on import, so the test output stays
# readable.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.tracking import MlflowClient  # noqa: E402
import pytest  # noqa: E402

from rlvr.app.cli import main  # noqa: E402
from rlvr.tracking.mlflow_view import (  # noqa: E402
    build_view_state,
    encode_view_state,
    write_view_tag,
)

# MLflow's SQLAlchemy store configures its mappers with a loader that SQLAlchemy
# deprecates. The warning is not this code's problem.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)

#: The section order and membership from the spec, written out by hand so a
#: change to the module fails here.
EXPECTED: dict[str, list[str]] = {
    "core": ["mean_reward", "pass_rate", "advantage_share"],
    "quality": [
        "pass_at_1/checkpoint",
        "pass_at_1/final",
        "mean_entropy",
        "truncation_rate",
        "group_spread",
        "group_sd",
    ],
    "optim": ["grad_norm", "update_norm"],
    "speed": [
        "step_seconds",
        "generation_seconds",
        "scoring_seconds",
        "scoring_wait_seconds",
        "update_seconds",
        "completion_tokens_per_second",
        "sequences_per_second",
    ],
    "data": [
        "prompts",
        "train_prompts",
        "parked_prompts",
        "sequences",
        "cache_hits",
        "mean_completion_tokens",
    ],
    "gpu": [
        "gpu_live_mib",
        "gpu_cached_mib",
        "gpu_fragmented_mib",
        "gpu_total_mib",
        "gpu_unaccounted_mib",
    ],
    "band": [
        "band/very_easy/pass_rate",
        "band/medium/pass_rate",
        "band/hard/pass_rate",
        "band/very_hard/pass_rate",
    ],
    "stage": [
        "stage/very_easy/solved",
        "stage/very_easy/total",
        "stage/medium/solved",
        "stage/medium/total",
        "stage/hard/solved",
        "stage/hard/total",
        "stage/very_hard/solved",
        "stage/very_hard/total",
    ],
}

#: The keys the Chart view would add on its own. A deleted card blocks the add
#: and does not render, so the page stays at eight sections.
EXPECTED_HIDDEN = {
    "rlvr/heartbeat_seconds",
    "system/cpu_utilization_percentage",
    "system/disk_available_megabytes",
    "system/disk_usage_megabytes",
    "system/disk_usage_percentage",
    "system/network_receive_megabytes",
    "system/network_transmit_megabytes",
    "system/system_memory_usage_megabytes",
    "system/system_memory_usage_percentage",
    "system/gpu_0_memory_usage_percentage",
    "system/gpu_0_memory_usage_megabytes",
    "system/gpu_0_utilization_percentage",
    "system/gpu_0_power_usage_watts",
    "system/gpu_0_power_usage_percentage",
}


def test_build_view_state_puts_the_core_first() -> None:
    state = build_view_state()

    sections = state["compareRunSections"]
    assert [section["name"] for section in sections] == list(EXPECTED)
    assert all(section["display"] is True for section in sections)
    assert all(section["isReordered"] is True for section in sections)

    cards = [card for card in state["compareRunCharts"] if not card["deleted"]]
    section_ids = {section["name"]: section["uuid"] for section in sections}
    for name, metrics in EXPECTED.items():
        found = [
            card["metricKey"]
            for card in cards
            if card["metricSectionId"] == section_ids[name]
        ]
        assert found == metrics
    assert len(cards) == sum(len(metrics) for metrics in EXPECTED.values()) == 41

    core_id = section_ids["core"]
    core = [card for card in cards if card["metricSectionId"] == core_id]
    assert all(card["type"] == "LINE" for card in core)
    assert all(card["xAxisKey"] == "step" for card in core)
    assert all(card["yAxisKey"] == "metric" for card in core)
    assert [card["range"] for card in core] == [
        {"yMin": -1.0, "yMax": 1.1},
        {"yMin": 0.0, "yMax": 1.05},
        {"yMin": 0.0, "yMax": 1.05},
    ]

    # A chart outside core keeps the auto range.
    rest = [card for card in cards if card["metricSectionId"] != core_id]
    assert rest
    assert all("range" not in card for card in rest)

    assert state["searchFilter"] == "tags.`rlvr/run_dir` != ''"

    hidden = [card for card in state["compareRunCharts"] if card["deleted"]]
    assert {card["metricKey"] for card in hidden} == EXPECTED_HIDDEN
    assert all(card["deleted"] is True for card in hidden)
    assert all(card["type"] == "LINE" for card in hidden)
    assert all(card["selectedMetricKeys"] == [card["metricKey"]] for card in hidden)
    assert {card["metricSectionId"] for card in hidden} <= set(section_ids.values())


def test_encode_view_state_matches_the_mlflow_format() -> None:
    state = build_view_state()

    envelope = json.loads(encode_view_state(state))

    assert set(envelope) == {"name", "createdAt", "updatedAt", "state"}
    assert isinstance(envelope["name"], str)
    assert isinstance(envelope["createdAt"], int)
    assert isinstance(envelope["updatedAt"], int)
    assert envelope["state"].startswith("deflate;")

    payload = base64.b64decode(envelope["state"].removeprefix("deflate;"))
    assert json.loads(zlib.decompress(payload)) == state

    # Random bytes do not compress, so this view is well over 20000 characters.
    state["padding"] = os.urandom(20000).hex()
    with pytest.raises(ValueError):
        encode_view_state(state)


def test_write_view_tag_replaces_the_tag(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    write_view_tag(runs_root)
    first = MlflowClient().get_experiment_by_name("rlvr")
    assert first is not None
    envelope = json.loads(first.tags["mlflow.sharedViewState.rlvr-study"])
    state = json.loads(
        zlib.decompress(base64.b64decode(envelope["state"].removeprefix("deflate;")))
    )
    assert state == build_view_state()

    write_view_tag(runs_root)
    second = MlflowClient().get_experiment_by_name("rlvr")
    assert second is not None
    keys = [key for key in second.tags if key.startswith("mlflow.sharedViewState.")]
    assert keys == ["mlflow.sharedViewState.rlvr-study"]
    assert second.experiment_id == first.experiment_id


def test_saved_view_command_writes_the_tag_and_prints_the_route(
    tmp_path: Path, capsys
) -> None:
    runs_root = tmp_path / "runs"

    assert main(["saved-view", "--runs-root", str(runs_root)]) == 0

    out = capsys.readouterr().out
    experiment = MlflowClient().get_experiment_by_name("rlvr")
    assert experiment is not None
    envelope = json.loads(experiment.tags["mlflow.sharedViewState.rlvr-study"])
    state = json.loads(
        zlib.decompress(base64.b64decode(envelope["state"].removeprefix("deflate;")))
    )
    assert state == build_view_state()

    assert out.strip().splitlines()[-1] == (
        f"#/experiments/{experiment.experiment_id}/runs"
        "?viewStateShareKey=rlvr-study&compareRunsMode=CHART"
    )

    # A second run replaces the tag, so the store keeps one saved view.
    assert main(["saved-view", "--runs-root", str(runs_root)]) == 0
    again = MlflowClient().get_experiment_by_name("rlvr")
    assert again is not None
    keys = [key for key in again.tags if key.startswith("mlflow.sharedViewState.")]
    assert keys == ["mlflow.sharedViewState.rlvr-study"]
