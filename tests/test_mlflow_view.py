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

from rlvr.mlflow_view import (  # noqa: E402
    TAG_KEY,
    build_view_state,
    encode_view_state,
    write_view_tag,
)

# MLflow's SQLAlchemy store configures its mappers with a loader that SQLAlchemy
# deprecates. The warning is not this code's problem.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)

CORE = ["mean_reward", "pass_rate", "advantage_share"]
SECTIONS = ["core", "quality", "optim", "speed", "data", "gpu"]


def test_build_view_state_puts_the_core_first() -> None:
    state = build_view_state()

    sections = state["compareRunSections"]
    assert [section["name"] for section in sections] == SECTIONS
    assert all(section["display"] is True for section in sections)
    assert all(section["isReordered"] is True for section in sections)

    core_id = sections[0]["uuid"]
    cards = state["compareRunCharts"]
    core = [card for card in cards if card["metricSectionId"] == core_id]
    assert [card["metricKey"] for card in core] == CORE
    assert all(card["type"] == "LINE" for card in core)
    assert all(card["xAxisKey"] == "step" for card in core)
    assert all(card["yAxisKey"] == "metric" for card in core)
    assert [card["range"] for card in core] == [
        {"yMin": -1.0, "yMax": 1.0},
        {"yMin": 0.0, "yMax": 1.0},
        {"yMin": 0.0, "yMax": 1.0},
    ]

    # A chart outside core keeps the auto range.
    rest = [card for card in cards if card["metricSectionId"] != core_id]
    assert rest
    assert all("range" not in card for card in rest)

    assert state["searchFilter"] == "tags.`rlvr/run_dir` != ''"


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
    envelope = json.loads(first.tags[TAG_KEY])
    state = json.loads(
        zlib.decompress(base64.b64decode(envelope["state"].removeprefix("deflate;")))
    )
    assert state == build_view_state()

    write_view_tag(runs_root)
    second = MlflowClient().get_experiment_by_name("rlvr")
    assert second is not None
    keys = [key for key in second.tags if key.startswith("mlflow.sharedViewState.")]
    assert keys == [TAG_KEY]
    assert second.experiment_id == first.experiment_id
