"""A saved MLflow view for the study page.

MLflow keeps a saved view as an experiment tag on the `rlvr` experiment. The
tag holds the chart sections, the chart order and the fixed y ranges. The page
reads the tag named by the URL, so every browser shows the same layout.

`build_view_state` returns the plain dict MLflow keeps. `encode_view_state`
wraps it in the envelope the UI reads: a JSON object whose `state` is
`deflate;` plus base64 of a zlib stream. `write_view_tag` writes the tag to
the store. `build_route` names the hash route that opens the view.

The three headline charts sit in the core section in a fixed order. Their y
ranges are fixed, so a low pass rate looks low. Every other chart keeps the
auto range. The search filter keeps the parent runs and drops the GPU child.
The view also carries deleted cards for the heartbeat and the system metrics,
so the Chart view does not add sections for them.
"""

from __future__ import annotations

import base64
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from .memory import MEMORY_METRIC_FIELDS
from .mlflow_log import ARTIFACTS_NAME, DATABASE_NAME, EXPERIMENT_NAME
from .stages import BANDS

__all__ = [
    "TAG_KEY",
    "build_route",
    "build_view_state",
    "encode_view_state",
    "write_view_tag",
]

#: The share key is the tag name the UI strips and then reads back from the
#: URL. A second write replaces the same tag, because tag keys are unique.
SHARE_KEY = "rlvr-study"
TAG_KEY = f"mlflow.sharedViewState.{SHARE_KEY}"

#: The name the UI shows in the saved view list.
VIEW_NAME = "rlvr study"

#: The prefix `textCompressDeflate` writes in the MLflow UI.
DEFLATE_PREFIX = "deflate;"

#: The server caps an experiment tag value at this many characters.
TAG_LIMIT = 20000

#: The parent run carries `rlvr/run_dir`. The GPU child does not.
FILTER = "tags.`rlvr/run_dir` != ''"

#: Keys the Chart view would add on its own. It adds a chart for every metric
#: key missing from the view, so a deleted card blocks the add. A deleted card
#: does not render, so the visible page stays at eight sections.
HIDDEN_METRICS = (
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
)

#: Section order on the page. The UI keeps this order because each section
#: sets `isReordered`.
_SECTIONS = (
    "core",
    "quality",
    "optim",
    "speed",
    "data",
    "gpu",
    "band",
    "stage",
)

#: The three headline charts, in order. Their ranges are fixed below.
_CORE = ("mean_reward", "pass_rate", "advantage_share")

_CORE_RANGES: dict[str, tuple[float, float]] = {
    "mean_reward": (-1.0, 1.0),
    "pass_rate": (0.0, 1.0),
    "advantage_share": (0.0, 1.0),
}


def _stage_metrics() -> tuple[str, ...]:
    """The stage solved and total series, one pair per band."""
    names: list[str] = []
    for band in BANDS:
        slug = band.replace(" ", "_")
        names.extend((f"stage/{slug}/solved", f"stage/{slug}/total"))
    return tuple(names)


def _band_metrics() -> tuple[str, ...]:
    """The cumulative pass rate series, one per band."""
    return tuple(f"band/{band.replace(' ', '_')}/pass_rate" for band in BANDS)


#: Every metric the mirror writes, placed in one section. The names match the
#: store, so old runs and the run folder stay compatible.
_SECTION_METRICS: dict[str, tuple[str, ...]] = {
    "core": _CORE,
    "quality": (
        "pass_at_1/checkpoint",
        "pass_at_1/final",
        "mean_entropy",
        "truncation_rate",
        "group_spread",
        "group_sd",
    ),
    "optim": (
        "grad_norm",
        "update_norm",
    ),
    "speed": (
        "step_seconds",
        "generation_seconds",
        "scoring_seconds",
        "scoring_wait_seconds",
        "update_seconds",
        "completion_tokens_per_second",
        "sequences_per_second",
    ),
    "data": (
        "prompts",
        "train_prompts",
        "parked_prompts",
        "sequences",
        "cache_hits",
        "mean_completion_tokens",
    ),
    "gpu": MEMORY_METRIC_FIELDS,
    "band": _band_metrics(),
    "stage": _stage_metrics(),
}


def _uuid(kind: str, name: str) -> str:
    """A stable id, so a rebuild of the view keeps the same layout."""
    return str(uuid5(NAMESPACE_URL, f"rlvr-study:{kind}:{name}"))


def _section(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "uuid": _uuid("section", name),
        "display": True,
        "isReordered": True,
        "columns": 3,
        "cardHeight": 360,
    }


def _card(metric: str, section_id: str) -> dict[str, Any]:
    card: dict[str, Any] = {
        "type": "LINE",
        "uuid": _uuid("chart", metric),
        "metricSectionId": section_id,
        "metricKey": metric,
        "selectedMetricKeys": [metric],
        "displayName": metric,
        "xAxisKey": "step",
        "yAxisKey": "metric",
        "isGenerated": False,
        "deleted": False,
        "runsCountToCompare": 10,
    }
    bounds = _CORE_RANGES.get(metric)
    if bounds is not None:
        # Only the y range is fixed. A missing x range stays auto.
        card["range"] = {"yMin": bounds[0], "yMax": bounds[1]}
    return card


def _hidden_card(metric: str, section_id: str) -> dict[str, Any]:
    """A card the UI keeps and does not render. It blocks the auto-add."""
    return {
        "type": "LINE",
        "uuid": _uuid("hidden", metric),
        "metricSectionId": section_id,
        "metricKey": metric,
        "selectedMetricKeys": [metric],
        "displayName": metric,
        "xAxisKey": "step",
        "yAxisKey": "metric",
        "isGenerated": False,
        "deleted": True,
        "runsCountToCompare": 10,
    }


def build_view_state() -> dict[str, Any]:
    """The chart sections, the chart order, the ranges and the filter."""
    sections = [_section(name) for name in _SECTIONS]
    section_ids = {section["name"]: section["uuid"] for section in sections}
    cards = [
        _card(metric, section_ids[name])
        for name in _SECTIONS
        for metric in _SECTION_METRICS[name]
    ]
    # The hidden cards point at a real section, because the UI resolves the id
    # before it drops the card.
    hidden = [_hidden_card(metric, section_ids["core"]) for metric in HIDDEN_METRICS]
    return {
        "compareRunCharts": cards + hidden,
        "compareRunSections": sections,
        "isAccordionReordered": True,
        "autoRefreshEnabled": True,
        "globalLineChartConfig": {
            "xAxisKey": "step",
            "lineSmoothness": 0,
            "selectedXAxisMetricKey": "",
        },
        "hideEmptyCharts": True,
        "searchFilter": FILTER,
    }


def encode_view_state(state: Mapping[str, Any]) -> str:
    """The tag value: the envelope with a `deflate;` state.

    The state is zlib compressed and base64 encoded, which is what
    `textCompressDeflate` does in the MLflow UI. The envelope length is checked
    before the store call, so an oversized view fails loudly.
    """
    raw = json.dumps(state, separators=(",", ":"), sort_keys=True).encode()
    payload = base64.b64encode(zlib.compress(raw)).decode("ascii")
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    envelope = json.dumps(
        {
            "name": VIEW_NAME,
            "createdAt": now,
            "updatedAt": now,
            "state": f"{DEFLATE_PREFIX}{payload}",
        },
        separators=(",", ":"),
    )
    if len(envelope) > TAG_LIMIT:
        raise ValueError(
            f"the saved view is {len(envelope)} characters, "
            f"over the {TAG_LIMIT} tag limit"
        )
    return envelope


def write_view_tag(runs_root: str | Path) -> str:
    """Write the saved view to the `rlvr` experiment and return its id.

    The experiment is created when the store does not have it yet. A second
    call replaces the same tag, so reruns do not pile up views. The id names the
    route, so the caller does not look the experiment up a second time.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    root = Path(runs_root)
    envelope = encode_view_state(build_view_state())
    root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{root / DATABASE_NAME}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=str(root / ARTIFACTS_NAME),
        )
    else:
        experiment_id = experiment.experiment_id
    client.set_experiment_tag(experiment_id, TAG_KEY, envelope)
    return experiment_id


def build_route(experiment_id: str) -> str:
    """The hash route that opens the saved view on the experiment Chart view.

    The view mode belongs in the URL, not in the tag. The share key is the tag
    suffix, which is what the UI reads.
    """
    return (
        f"#/experiments/{experiment_id}/runs"
        f"?viewStateShareKey={SHARE_KEY}&compareRunsMode=CHART"
    )
