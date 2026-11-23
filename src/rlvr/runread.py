"""A live progress dashboard served over a local HTTP server.

The dashboard reads the same run folders the summary reads, but never fails on
a run that is mid-write. It serves two resources:

    GET /          one HTML page with the React bundle inlined
    GET /data.json the JSON the page polls every half second

The page is a monitoring tool, not a report. One section per arm, one KPI strip
per section, and a grid of charts. Distributions come from the tail of
`rollouts.jsonl`, so a long run does not re-read the whole file on every poll.
GPU samples come from `nvidia-smi`, at most one every two seconds on the
`/data.json` route, and the last 120 samples are kept in memory for the GPU
chart. The per step memory split (live, cached, fragmented, device total,
unaccounted) comes from the metrics rows the trainer writes, so the memory
panel works on a machine with no GPU.

The page is built from the Vite app in `dashboard/` and committed as a single
HTML file in the `rlvr/web` package data. It fetches nothing from the network,
so the dashboard works on a host with no internet access.

The server binds to 127.0.0.1 and stops on Ctrl+C. Nothing here reads secrets
or environment variables.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from copy import deepcopy
from threading import RLock
from collections import OrderedDict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import urlsplit

from .dashboard_io import _all_rollout_records, _read_json, _read_jsonl, _read_tail_lines
from .memory import MEMORY_METRIC_FIELDS
from .stages import BANDS, UNSTAGED_STAGE

__all__ = ["collect_dashboard", "render_html", "sample_gpu", "serve_dashboard"]

METRICS_NAME = "metrics.jsonl"
EVAL_NAME = "eval.json"
META_NAME = "meta.json"
ROLLOUTS_NAME = "rollouts.jsonl"

#: The control file a run polls between steps, and the log of the changes it
#: applied. Both are written by the training loop in the run folder.
CONTROL_NAME = "control.json"
CONTROL_EVENTS_NAME = "control-events.jsonl"

#: Question files that carry prompt text. They live in `data/` in the repo and
#: are optional: the questions view works from prompt ids alone when they are
#: missing.
QUESTION_SOURCES: tuple[tuple[str, str], ...] = (
    ("mbpp", "mbpp_full.jsonl"),
    ("synthetic", "synthetic_easy.jsonl"),
)

#: Rollouts per prompt in a step when meta.json does not record the count.
DEFAULT_ROLLOUTS_PER_PROMPT = 8

#: A metrics file not touched in this many seconds counts as stopped, unless a
#: final evaluation is already on disk. One step is a vLLM batch plus a test run
#: per rollout, which took about a minute on the 3090 and longer on wide batches,
#: so a one minute window reported finished runs as stopped.
STALE_SECONDS = 900.0

#: Seconds after the last metrics write before a run with no matching `rlvr run`
#: process counts as stopped. A crashed resume leaves a fresh metrics file
#: behind, so the file age alone reported a dead run as running for fifteen
#: minutes. The grace period covers a process that has just exited and a host
#: where the process list cannot be read.
PROCESS_GRACE_SECONDS = 120.0

#: Where to look for nvidia-smi when PATH does not carry it. A tmux server
#: started from a different shell can have a PATH without the driver tools even
#: though the binary is installed.
NVIDIA_SMI_PATHS: tuple[str, ...] = (
    "/usr/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
)

#: How many bytes of rollouts.jsonl to read from the tail for distributions.
ROLLOUT_TAIL_BYTES = 4 * 1024 * 1024

#: Number of bins for the reward histogram.
REWARD_BINS = 20

#: The reward is clipped to this range in the trainer, so the histogram covers
#: it. Negative rewards are common: a truncated answer scores -1.0.
REWARD_MIN = -1.0
REWARD_MAX = 1.0

#: A GPU sample younger than this many seconds is reused instead of sampled
#: again, so a page that polls fast does not spawn nvidia-smi every request.
GPU_SAMPLE_SECONDS = 2.0

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

SERIES_FIELDS: tuple[str, ...] = (
    "mean_reward",
    "pass_rate",
    "mean_entropy",
    "grad_norm",
    "update_norm",
    "mean_completion_tokens",
    "truncation_rate",
    "cache_hits",
    "prompts",
    "sequences",
) + MEMORY_METRIC_FIELDS + (
    "generation_seconds",
    "scoring_seconds",
    "scoring_wait_seconds",
    "update_seconds",
    "step_seconds",
    "completion_tokens_per_second",
    "sequences_per_second",
    "train_prompts",
    "parked_prompts",
    "advantage_share",
    "group_spread",
    "group_sd",
)

#: Fields that are whole counts and are read as integers, not floats.
COUNT_SERIES_FIELDS: tuple[str, ...] = (
    "cache_hits",
    "prompts",
    "sequences",
    "train_prompts",
    "parked_prompts",
)

LATEST_FIELDS: tuple[str, ...] = SERIES_FIELDS + ("rollouts",)

#: The last 120 GPU samples, one per data request.
_gpu_history: deque[dict[str, Any]] = deque(maxlen=120)
_gpu_lock = RLock()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _parse_timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.timestamp()


def _started_label(value: str | None) -> str | None:
    """A run timestamp as a readable UTC label, or the raw value if unparsable."""
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M:%SZ")


def _read_meta(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / META_NAME)


def _read_eval(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / EVAL_NAME)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def _rollout_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rollout records from the tail of a rollouts.jsonl file."""
    for line in _read_tail_lines(path, ROLLOUT_TAIL_BYTES):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _looks_like_run(path: Path) -> bool:
    return path.is_dir() and (
        (path / METRICS_NAME).is_file() or (path / META_NAME).is_file()
    )


def _latest_run_in(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    candidates = [child for child in folder.iterdir() if _looks_like_run(child)]
    if not candidates:
        return None
    return max(candidates, key=lambda child: child.name)


def _resolve_paths(paths: Iterable[str | Path]) -> list[tuple[str, Path | None]]:
    """Turn run folders, arm folders, and runs roots into (arm, run) targets.

    A run folder keeps its parent as the arm. A runs root expands to each arm's
    latest run. An arm with no run yet yields a `None` run, which the page shows
    as waiting.
    """
    targets: list[tuple[str, Path | None]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if _looks_like_run(path):
            targets.append((path.parent.name, path))
            continue
        if not path.is_dir():
            continue
        children = sorted(child for child in path.iterdir() if child.is_dir())
        run_children = [child for child in children if _looks_like_run(child)]
        if run_children:
            latest = max(run_children, key=lambda child: child.name)
            targets.append((path.name, latest))
        else:
            for child in children:
                targets.append((child.name, _latest_run_in(child)))
    return targets


def _schedule_step_prompts(meta: Mapping[str, Any]) -> int | None:
    """The scheduled step size recorded by a pooled run, or None."""
    schedule = meta.get("schedule")
    if not isinstance(schedule, Mapping):
        return None
    return _positive_int(schedule.get("step_prompts"))


def _pool_policy(meta: Mapping[str, Any]) -> str | None:
    """The pool re-screen policy the run used, or None for an unpooled run.

    `pool_policy` is the current key. A meta file written before it was
    promoted to the top level still carries the policy inside `schedule`, so
    that value is read as the fallback.
    """
    policy = _string(meta.get("pool_policy"))
    if policy is not None:
        return policy
    schedule = meta.get("schedule")
    if isinstance(schedule, Mapping):
        return _string(schedule.get("policy"))
    return None


def _pool_train_size(meta: Mapping[str, Any]) -> int | None:
    """The measured train pool count recorded by a pooled run, or None."""
    pools = meta.get("pools")
    if not isinstance(pools, Mapping):
        return None
    return _positive_int(pools.get("train"))


def _real_target_steps(meta: Mapping[str, Any], config: Mapping[str, Any]) -> int | None:
    """Epochs times the train size, divided into batches of prompts.

    A pooled run records its scheduled step size, which replaces the adaptive
    starting batch. Its train size is the measured pool, not the config split.
    A meta file written before batching has no start size, so it counts one
    prompt per step and keeps the old figure.
    """
    epochs: int | None = None
    batch_prompts = 1
    trainer = meta.get("trainer")
    if isinstance(trainer, dict):
        epochs = _positive_int(trainer.get("epochs"))
        batch_prompts = _positive_int(trainer.get("batch_start_prompts")) or 1
    if epochs is None:
        training = config.get("training")
        if isinstance(training, dict):
            epochs = _positive_int(training.get("epochs"))
    data = config.get("data")
    if epochs is None or not isinstance(data, dict):
        return None
    schedule_step = _schedule_step_prompts(meta)
    train_size = _pool_train_size(meta) if schedule_step is not None else None
    if train_size is None:
        train_size = _positive_int(data.get("train_size"))
    if schedule_step is not None:
        batch_prompts = schedule_step
    if train_size is None:
        return None
    return -(-(epochs * train_size) // batch_prompts)


def _fake_target_steps(config: Mapping[str, Any]) -> int | None:
    """The step count the fake path runs, from the [fake] table."""
    fake = config.get("fake")
    if not isinstance(fake, dict):
        return None
    return _positive_int(fake.get("steps"))


def _target_steps(meta: Mapping[str, Any]) -> int | None:
    """Target steps derivable from meta.json, or None when it is not recorded.

    `mode` picks the formula. A fake run stops after `[fake] steps`, so it must
    not be shown the real epoch budget. Meta files written before `mode` existed
    carry no mode and fall back to the real formula.
    """
    config = meta.get("config")
    if not isinstance(config, dict):
        return None
    if meta.get("mode") == "fake":
        target = _fake_target_steps(config)
    else:
        target = _real_target_steps(meta, config)
    if target is None:
        return None
    budget = config.get("budget")
    if isinstance(budget, dict):
        max_steps = _positive_int(budget.get("max_steps"))
        if max_steps is not None:
            target = min(target, max_steps)
    return target


def _prompt_sample_target(meta: Mapping[str, Any]) -> int | None:
    """The number of prompt presentations the run plans to make.

    A pooled run has a fixed step size, so the plan is the target step count
    times the scheduled prompts per step. Without a schedule, one step samples a
    growing batch of prompts, so the step count depends on the batching policy.
    The number of prompts the run has to present does not, which makes it the
    honest progress denominator for a real run.
    """
    if meta.get("mode") == "fake":
        return None
    schedule_step = _schedule_step_prompts(meta)
    if schedule_step is not None:
        target = _target_steps(meta)
        if target is None:
            return None
        return target * schedule_step
    epochs: int | None = None
    trainer = meta.get("trainer")
    if isinstance(trainer, Mapping):
        epochs = _positive_int(trainer.get("epochs"))
    config = meta.get("config")
    if epochs is None and isinstance(config, Mapping):
        training = config.get("training")
        if isinstance(training, Mapping):
            epochs = _positive_int(training.get("epochs"))
    if epochs is None or not isinstance(config, Mapping):
        return None
    data = config.get("data")
    if not isinstance(data, Mapping):
        return None
    train_size = _positive_int(data.get("train_size"))
    if train_size is None:
        return None
    return epochs * train_size


def _start_batch_prompts(meta: Mapping[str, Any]) -> int | None:
    """The prompts per step the step target used, when it is recorded.

    A pooled run uses its scheduled step size. Without a schedule this is the
    adaptive starting batch.
    """
    schedule_step = _schedule_step_prompts(meta)
    if schedule_step is not None:
        return schedule_step
    trainer = meta.get("trainer")
    if isinstance(trainer, Mapping):
        return _positive_int(trainer.get("batch_start_prompts"))
    return None


def _entropy_every_steps(meta: Mapping[str, Any]) -> int | None:
    """How often entropy is measured, so a dash on the tile can say why."""
    trainer = meta.get("trainer")
    if isinstance(trainer, Mapping):
        value = _positive_int(trainer.get("entropy_every_steps"))
        if value is not None:
            return value
    config = meta.get("config")
    if isinstance(config, Mapping):
        training = config.get("training")
        if isinstance(training, Mapping):
            return _positive_int(training.get("entropy_every_steps"))
    return None


def _steps_done(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    step = _integer(rows[-1].get("step"))
    return step if step is not None and step > 0 else len(rows)


def _prompt_samples(rows: list[dict[str, Any]]) -> int:
    """The prompts the run has presented, summed over the metrics rows."""
    total = 0
    for row in rows:
        value = _integer(row.get("prompts"))
        if value is not None:
            total += value
    return total


def _last_number(rows: list[dict[str, Any]], name: str) -> float | None:
    """The last present numeric value of a field, or None.

    Entropy is measured only every `entropy_every_steps` training steps, so the newest row
    often has no value. The KPI shows the last measured value instead of
    flickering back to a dash.
    """
    for row in reversed(rows):
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def _latest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {name: None for name in LATEST_FIELDS}
    last = rows[-1]
    result: dict[str, Any] = {}
    for name in SERIES_FIELDS:
        result[name] = (
            _integer(last.get(name))
            if name in COUNT_SERIES_FIELDS
            else _number(last.get(name))
        )
    result["mean_entropy"] = _last_number(rows, "mean_entropy")
    result["rollouts"] = _integer(last.get("rollouts"))
    return result


def _series(rows: list[dict[str, Any]]) -> dict[str, Any]:
    series: dict[str, Any] = {"steps": []}
    for name in SERIES_FIELDS:
        series[name] = []
    for row in rows:
        series["steps"].append(_integer(row.get("step")))
        for name in SERIES_FIELDS:
            value = (
                _integer(row.get(name))
                if name in COUNT_SERIES_FIELDS
                else _number(row.get(name))
            )
            series[name].append(value)
    return series


def _raw_evaluations(eval_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    evaluations = eval_payload.get("evaluations")
    if not isinstance(evaluations, list):
        return []
    return [dict(point) for point in evaluations if isinstance(point, dict)]


def _evaluations(eval_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for point in _raw_evaluations(eval_payload):
        result.append(
            {
                "step": _integer(point.get("step")),
                "kind": _string(point.get("kind")) or "",
                "stage": _string(point.get("stage")),
                "count": _integer(point.get("count")),
                "pass_at_1": _number(point.get("pass_at_1")),
            }
        )
    return result


def _pass_at_1_series(eval_payload: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {"steps": [], "values": []}
    final: dict[str, Any] = {"steps": [], "values": []}
    for point in _raw_evaluations(eval_payload):
        kind = point.get("kind")
        step = _integer(point.get("step"))
        value = _number(point.get("pass_at_1"))
        if kind == "checkpoint":
            checkpoint["steps"].append(step)
            checkpoint["values"].append(value)
        elif kind == "final":
            final["steps"].append(step)
            final["values"].append(value)
    return {"checkpoint": checkpoint, "final": final}


def _reward_histogram(rewards: list[float]) -> dict[str, Any]:
    """Count rewards into fixed bins spanning -1 to 1, both edges inclusive."""
    span = REWARD_MAX - REWARD_MIN
    counts = [0] * REWARD_BINS
    for reward in rewards:
        bounded = min(max(reward, REWARD_MIN), REWARD_MAX)
        index = min(int((bounded - REWARD_MIN) / span * REWARD_BINS), REWARD_BINS - 1)
        counts[index] += 1
    bins = [REWARD_MIN + index * span / REWARD_BINS for index in range(REWARD_BINS)]
    return {"bins": bins, "counts": counts}


def _tests_passed_histogram(passed_counts: list[int]) -> dict[str, Any]:
    if not passed_counts:
        return {"bins": [], "counts": []}
    highest = max(passed_counts)
    counts = [0] * (highest + 1)
    for passed in passed_counts:
        counts[passed] += 1
    return {"bins": list(range(highest + 1)), "counts": counts}


def _band_pass_rates(run_dir: Path) -> list[dict[str, Any]]:
    """Full pass rate per stage value, over the whole rollout file.

    A rollout counts as passed only when every test passed, matching the
    training pool rule. The four fixed bands appear in order first, then any
    other stage value the run folder carries, including `unstaged`.
    """
    counts: dict[str, int] = {}
    passed: dict[str, int] = {}
    for record in _all_rollout_records(run_dir / ROLLOUTS_NAME):
        band = _string(record.get("stage"))
        if band is None:
            continue
        counts[band] = counts.get(band, 0) + 1
        passed_count = _integer(record.get("tests_passed"))
        total = _integer(record.get("tests_total"))
        if (
            passed_count is not None
            and total is not None
            and total > 0
            and passed_count == total
        ):
            passed[band] = passed.get(band, 0) + 1

    ordered = [band for band in BANDS if band in counts]
    ordered += sorted(band for band in counts if band not in BANDS)
    return [
        {
            "band": band,
            "rollouts": counts[band],
            "pass_rate": passed.get(band, 0) / counts[band],
        }
        for band in ordered
    ]


def _stage_progress(
    run_dir: Path, meta: Mapping[str, Any], dropped_by_band: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """How much of the current stage value has been solved, per step.

    A prompt counts as solved in a step only when every rollout of that prompt
    in that step passed, which is the training pool rule. The total for a stage
    is its starting training pool from `meta.json`, so the percentage is stable
    even though the pool shrinks.

    A prompt dropped through the control file can never be solved, so it is
    removed from that band's total. Without this a band with dropped prompts
    tops out below 100% forever. The caller passes counts of dropped prompts
    that were never solved, one per band.

    The whole rollout file is streamed, because a solved prompt can sit far
    before the tail. The series keeps one point per training step, and carries
    the band with it, because a run that moves to a new band restarts at zero.

    A run with no curriculum has one pool. Its total is the `train` count from
    `meta.json`, under the `unstaged` band name, so the chart still renders.
    """
    curriculum = meta.get("curriculum")
    totals: dict[str, int] = {}
    if isinstance(curriculum, Mapping):
        stages = curriculum.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, Mapping):
                    continue
                name = _string(stage.get("name"))
                count = _integer(stage.get("train"))
                if name is not None and count:
                    totals[name] = count

    pools = meta.get("pools")
    if isinstance(pools, Mapping):
        count = _integer(pools.get("train"))
        if count:
            totals[UNSTAGED_STAGE] = count

    dropped_counts = dropped_by_band or {}
    solved: dict[str, set[str]] = {}
    points: list[dict[str, Any]] = []

    def close_step(step: int, band: str, prompts: dict[str, list[bool]]) -> None:
        for prompt_id, outcomes in prompts.items():
            if outcomes and all(outcomes):
                solved.setdefault(band, set()).add(prompt_id)
        starting_total = totals.get(band)
        if not starting_total:
            return
        dropped = min(dropped_counts.get(band, 0), starting_total)
        total = starting_total - dropped
        if not total:
            return
        count = len(solved.get(band, ()))
        points.append(
            {
                "step": step,
                "band": band,
                "solved": count,
                "total": total,
                "dropped": dropped,
                "percent": min(count / total, 1.0),
            }
        )

    # Rollouts are appended step by step, so one step of prompt outcomes is
    # enough memory. The solved sets hold prompt ids only, one entry per prompt.
    current: tuple[int, str] | None = None
    prompts: dict[str, list[bool]] = {}
    for record in _all_rollout_records(run_dir / ROLLOUTS_NAME):
        step = _integer(record.get("step"))
        band = _string(record.get("stage"))
        prompt_id = _string(record.get("prompt_id"))
        if step is None or band is None or prompt_id is None:
            continue
        passed = _integer(record.get("tests_passed"))
        total = _integer(record.get("tests_total"))
        full = passed is not None and total is not None and total > 0 and passed == total
        key = (step, band)
        if key != current:
            if current is not None:
                close_step(current[0], current[1], prompts)
            current = key
            prompts = {}
        prompts.setdefault(prompt_id, []).append(full)

    if current is not None:
        close_step(current[0], current[1], prompts)

    latest = points[-1] if points else None
    return {
        "band": latest["band"] if latest else None,
        "solved": latest["solved"] if latest else 0,
        "total": latest["total"] if latest else 0,
        "dropped": latest["dropped"] if latest else 0,
        "percent": latest["percent"] if latest else None,
        "points": points,
    }


def _rollouts_per_prompt(meta: Mapping[str, Any]) -> int:
    """The number of completions the trainer samples for one prompt in a step."""
    trainer = meta.get("trainer")
    if isinstance(trainer, Mapping):
        count = _positive_int(trainer.get("generations_per_prompt"))
        if count is not None:
            return count
    config = meta.get("config")
    if isinstance(config, Mapping):
        training = config.get("training")
        if isinstance(training, Mapping):
            count = _positive_int(training.get("generations_per_prompt"))
            if count is not None:
                return count
    return DEFAULT_ROLLOUTS_PER_PROMPT


def _question_text_files() -> list[tuple[str, Path]]:
    """Candidate question files next to the repo or the working directory."""
    roots: list[Path] = []
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if root not in roots:
            roots.append(root)
    files: list[tuple[str, Path]] = []
    for root in roots:
        for prefix, name in QUESTION_SOURCES:
            path = root / "data" / name
            if path.is_file():
                files.append((prefix, path))
    return files


def _question_texts(needed: set[str]) -> dict[str, str]:
    """Prompt text for the given ids, or an empty map when files are absent.

    The files are streamed one line at a time and only the text for a prompt the
    run sampled is kept, so a large question file costs little memory.
    """
    if not needed:
        return {}
    texts: dict[str, str] = {}
    for prefix, path in _question_text_files():
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line in handle:
                if len(texts) == len(needed):
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                prompt_id = _string(row.get("prompt_id"))
                if prompt_id is None:
                    task_id = _integer(row.get("task_id"))
                    if task_id is not None:
                        prompt_id = f"{prefix}-{task_id}"
                if prompt_id is None or prompt_id not in needed:
                    continue
                text = _string(row.get("text"))
                if text is not None:
                    texts[prompt_id] = text
    return texts


def _control_drops(run_dir: Path) -> dict[str, Any]:
    """The prompts a run has stopped sampling, with the step each was dropped.

    `control.json` holds the list in force now. `control-events.jsonl` holds one
    line per applied change, so it is the only source for the step. A drop is
    permanent in the training loop, so an id that appears in the history stays
    dropped even if it is no longer in the current list. A missing file
    contributes nothing. An id with no matching event has a null step.
    """
    control = _read_json(run_dir / CONTROL_NAME)
    current_raw = control.get("drop")
    current: set[str] = set()
    if isinstance(current_raw, list):
        current = {item for item in current_raw if isinstance(item, str) and item}

    steps: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    for row in _read_jsonl(run_dir / CONTROL_EVENTS_NAME):
        if row.get("action") != "apply":
            continue
        dropped_raw = row.get("drop")
        if not isinstance(dropped_raw, list):
            continue
        prompt_ids = sorted(
            {item for item in dropped_raw if isinstance(item, str) and item}
        )
        if not prompt_ids:
            continue
        step = _integer(row.get("step"))
        events.append({"step": step, "prompt_ids": prompt_ids})
        if step is not None:
            for prompt_id in prompt_ids:
                if prompt_id not in steps or step < steps[prompt_id]:
                    steps[prompt_id] = step

    ids = sorted(current | set(steps))
    return {
        "ids": ids,
        "events": events,
        "steps": {prompt_id: steps[prompt_id] for prompt_id in ids if prompt_id in steps},
    }


def _questions(
    run_dir: Path, meta: Mapping[str, Any], dropped: Mapping[str, Any]
) -> dict[str, Any]:
    """One row per prompt, from a single streaming pass over rollouts.jsonl.

    A prompt is solved when every rollout of it in a step passed, matching the
    training pool rule. The step group is closed when the step number changes,
    so memory holds counters, not rollout records. A prompt named in the
    control file keeps its history and is marked dropped, so a solved prompt
    reads as solved, not as a failure.
    """
    dropped_ids = set(dropped.get("ids", ()))
    dropped_steps = dropped.get("steps") or {}
    expected = _rollouts_per_prompt(meta)
    counters: dict[str, dict[str, Any]] = {}
    group: dict[str, dict[str, int]] = {}
    current_step: int | None = None

    def close_group() -> None:
        for prompt_id, outcome in group.items():
            counter = counters.get(prompt_id)
            if counter is None:
                continue
            counter["steps"] += 1
            if outcome["count"] >= expected and outcome["passed"] == outcome["count"]:
                counter["full_pass_steps"] += 1

    for record in _all_rollout_records(run_dir / ROLLOUTS_NAME):
        step = _integer(record.get("step"))
        prompt_id = _string(record.get("prompt_id"))
        if step is None or prompt_id is None:
            continue
        if step != current_step:
            close_group()
            group = {}
            current_step = step

        counter = counters.get(prompt_id)
        if counter is None:
            counter = {
                "stage": None,
                "steps": 0,
                "rollouts": 0,
                "passed_rollouts": 0,
                "full_pass_steps": 0,
                "best_passed": None,
                "best_total": None,
                "best_ratio": -1.0,
                "last_error": None,
                "last_step": None,
            }
            counters[prompt_id] = counter

        stage = _string(record.get("stage"))
        if stage is not None:
            counter["stage"] = stage
        counter["last_step"] = step
        counter["rollouts"] += 1

        passed = _integer(record.get("tests_passed"))
        total = _integer(record.get("tests_total"))
        full = passed is not None and total is not None and total > 0 and passed == total
        if full:
            counter["passed_rollouts"] += 1
        if passed is not None and total is not None and total > 0:
            ratio = passed / total
            if ratio > counter["best_ratio"] or (
                ratio == counter["best_ratio"] and passed > (counter["best_passed"] or -1)
            ):
                counter["best_ratio"] = ratio
                counter["best_passed"] = passed
                counter["best_total"] = total

        error = _string(record.get("error_class"))
        if error is not None:
            counter["last_error"] = error

        outcome = group.setdefault(prompt_id, {"count": 0, "passed": 0})
        outcome["count"] += 1
        if full:
            outcome["passed"] += 1

    close_group()

    texts = _question_texts(set(counters))
    rows: list[dict[str, Any]] = []
    for prompt_id, counter in counters.items():
        rollouts = counter["rollouts"]
        pass_rate = counter["passed_rollouts"] / rollouts if rollouts else 0.0
        rows.append(
            {
                "prompt_id": prompt_id,
                "stage": counter["stage"],
                "steps": counter["steps"],
                "rollouts": rollouts,
                "full_pass_steps": counter["full_pass_steps"],
                "pass_rate": pass_rate,
                "best_passed": counter["best_passed"],
                "best_total": counter["best_total"],
                "last_error": counter["last_error"],
                "last_step": counter["last_step"],
                "solved": counter["full_pass_steps"] > 0,
                "dropped": prompt_id in dropped_ids,
                "dropped_step": dropped_steps.get(prompt_id),
                "text": texts.get(prompt_id),
            }
        )

    rows.sort(key=lambda row: (row["solved"], row["pass_rate"], -row["rollouts"], row["prompt_id"]))
    return {"total": len(rows), "rows": rows}


def _distributions(run_dir: Path) -> dict[str, Any]:
    rewards: list[float] = []
    passed_counts: list[int] = []
    for record in _rollout_records(run_dir / ROLLOUTS_NAME):
        reward = _number(record.get("reward"))
        if reward is not None:
            rewards.append(reward)
        passed = _integer(record.get("tests_passed"))
        if passed is not None:
            passed_counts.append(passed)
    return {
        "reward": _reward_histogram(rewards),
        "tests_passed": _tests_passed_histogram(passed_counts),
    }


def _metrics_age(metrics_path: Path, now: float) -> float | None:
    """Seconds since the metrics file was last written, or None."""
    try:
        return round(now - metrics_path.stat().st_mtime, 1)
    except OSError:
        return None


def _status(
    eval_payload: Mapping[str, Any],
    metrics_path: Path,
    now: float,
    alive: bool | None = None,
) -> str:
    """The run status, from the final eval, the metrics age, and the process.

    `alive` is True when a training process for the arm is present, False when
    the process list was read and no such process exists, and None when the
    list cannot be read. A live process keeps a run running through a long
    evaluation. A missing process only demotes a run after
    `PROCESS_GRACE_SECONDS`, so a just-exited process and an unreadable process
    list do not flip a fresh run to stopped.
    """
    if any(point.get("kind") == "final" for point in _raw_evaluations(eval_payload)):
        return "finished"
    if not metrics_path.is_file():
        return "waiting"
    try:
        mtime = metrics_path.stat().st_mtime
    except OSError:
        return "stopped"
    age = now - mtime
    if alive is True:
        return "running"
    if age <= STALE_SECONDS and (alive is not False or age <= PROCESS_GRACE_SECONDS):
        return "running"
    return "stopped"


def _process_arms() -> set[str] | None:
    """The arms that have a live `rlvr run` process, or None when unreadable.

    The trainer's command line carries `--arm <arm>`, and `run.sh` execs the
    entry point, so the process keeps that argument. Scanning `/proc` costs a
    few hundred small reads per poll. Only Linux hosts have it. A host without
    `/proc`, or one the dashboard cannot list, returns None so the caller falls
    back to the metrics age instead of guessing that nothing is running.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    try:
        entries = [entry for entry in proc.iterdir() if entry.name.isdigit()]
    except OSError:
        return None
    arms: set[str] = set()
    for entry in entries:
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"rlvr" not in raw:
            continue
        parts = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        for index, part in enumerate(parts):
            if part == "--arm" and index + 1 < len(parts):
                arms.add(parts[index + 1])
            elif part.startswith("--arm="):
                arms.add(part.split("=", 1)[1])
    return arms



def _waiting_run(arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "run_name": None,
        "run_dir": None,
        "started": None,
        "started_utc": None,
        "elapsed_seconds": None,
        "status": "waiting",
        "metrics_age_seconds": None,
        "steps_done": 0,
        "target_steps": None,
        "start_batch_prompts": None,
        "prompt_samples": 0,
        "prompt_sample_target": None,
        "entropy_every_steps": None,
        "pool_policy": None,
        "stage": None,
        "bands": [],
        "stage_progress": {
            "band": None,
            "solved": 0,
            "total": 0,
            "dropped": 0,
            "percent": None,
            "points": [],
        },
        "latest": {name: None for name in LATEST_FIELDS},
        "series": {"steps": []} | {name: [] for name in SERIES_FIELDS},
        "pass_at_1": {"checkpoint": {"steps": [], "values": []}, "final": {"steps": [], "values": []}},
        "evaluations": [],
        "distributions": {"reward": {"bins": [], "counts": []}, "tests_passed": {"bins": [], "counts": []}},
        "questions": {"total": 0, "rows": []},
        "dropped": {"ids": [], "events": [], "steps": {}},
    }


def _collect_run(
    arm_hint: str, run_dir: Path, now: float, live_arms: set[str] | None = None
) -> dict[str, Any]:
    meta = _read_meta(run_dir)
    eval_payload = _read_eval(run_dir)
    rows = _read_metrics(run_dir / METRICS_NAME)
    dropped = _control_drops(run_dir)

    arm = _string(meta.get("arm")) or _string(eval_payload.get("arm")) or arm_hint or run_dir.name
    timestamp = _string(meta.get("timestamp")) or _string(eval_payload.get("timestamp")) or run_dir.name
    started = _parse_timestamp(timestamp)
    elapsed = now - started if started is not None else None
    alive = None if live_arms is None else arm in live_arms

    questions = _questions(run_dir, meta, dropped)
    dropped_by_band: dict[str, int] = {}
    for row in questions["rows"]:
        if row["dropped"] and not row["solved"] and row["stage"] is not None:
            dropped_by_band[row["stage"]] = dropped_by_band.get(row["stage"], 0) + 1

    return {
        "arm": arm,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "started": timestamp,
        "started_utc": _started_label(timestamp),
        "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
        "status": _status(eval_payload, run_dir / METRICS_NAME, now, alive),
        "metrics_age_seconds": _metrics_age(run_dir / METRICS_NAME, now),
        "steps_done": _steps_done(rows),
        "target_steps": _target_steps(meta),
        "start_batch_prompts": _start_batch_prompts(meta),
        "prompt_samples": _prompt_samples(rows),
        "prompt_sample_target": _prompt_sample_target(meta),
        "entropy_every_steps": _entropy_every_steps(meta),
        "pool_policy": _pool_policy(meta),
        "stage": _string(rows[-1].get("stage")) if rows else None,
        "bands": _band_pass_rates(run_dir),
        "stage_progress": _stage_progress(run_dir, meta, dropped_by_band),
        "latest": _latest(rows),
        "series": _series(rows),
        "pass_at_1": _pass_at_1_series(eval_payload),
        "evaluations": _evaluations(eval_payload),
        "distributions": _distributions(run_dir),
        "questions": questions,
        "dropped": dropped,
    }


_GPU_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def _gpu_number(value: str) -> float | None:
    """An nvidia-smi CSV field as a number, or None when it is not one.

    nvidia-smi prints text tokens such as N/A, [N/A], [Not Supported] and
    [Unknown Error] for a field it cannot report, so anything that does not
    start with a number becomes None. A number with a unit still parses, so a
    format that keeps `%` or `MiB` does not silently zero the field. Zero is a
    real reading and stays zero.
    """
    text = value.strip()
    if not text:
        return None
    match = _GPU_NUMBER.match(text)
    if match is None:
        return None
    return _number(float(match.group(0)))


def _nvidia_smi() -> str:
    """The nvidia-smi binary by PATH, then by the usual install paths."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in NVIDIA_SMI_PATHS:
        if Path(candidate).is_file():
            return candidate
    return "nvidia-smi"


def _gpu_read() -> dict[str, Any] | None:
    """One nvidia-smi reading for every GPU, or None when nothing was read.

    One CSV row per GPU. Utilisation and temperature take the hottest GPU and
    memory is the sum over all of them, so a multi-GPU host reports the whole
    machine rather than an arbitrary card. A row with a field nvidia-smi cannot
    report keeps that field as None. A reading where every field is None is a
    failure, not a sample, so it is not recorded or counted.
    """
    command = [
        _nvidia_smi(),
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    rows: list[dict[str, float | None]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "utilization": _gpu_number(parts[0]),
                "memory_used": _gpu_number(parts[1]),
                "memory_total": _gpu_number(parts[2]),
                "temperature": _gpu_number(parts[3]),
            }
        )
    if not rows:
        return None

    def total(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            return None
        return float(sum(values))

    def highest(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            return None
        return float(max(values))

    sample = {
        "utilization": highest("utilization"),
        "memory_used": total("memory_used"),
        "memory_total": total("memory_total"),
        "temperature": highest("temperature"),
    }
    if all(value is None for value in sample.values()):
        return None
    return sample


def sample_gpu(now: float | None = None) -> None:
    """Sample nvidia-smi unless the last sample is still fresh.

    The sample is kept in the rolling history. A sample younger than
    `GPU_SAMPLE_SECONDS` is reused, so the 404 and HTML paths do not run
    nvidia-smi and a fast poll does not either. A failed read adds nothing and
    does not reset the last sample time, so the strip can show how long ago the
    last real reading was.
    """
    with _gpu_lock:
        current = time.time() if now is None else now
        if _gpu_history:
            age = current - _gpu_history[-1].get("t", 0.0)
            if age < GPU_SAMPLE_SECONDS:
                return
        sample = _gpu_read()
        if sample is not None:
            sample["t"] = current
            _gpu_history.append(sample)


def _gpu_state(now: float | None = None) -> dict[str, Any]:
    """The current sample, the history, and the age of the last reading.

    `age_seconds` is None when no reading has ever succeeded. The page shows
    that age next to the sample count, so a sampler that has stopped returning
    readings is visible instead of looking like a quiet GPU.
    """
    with _gpu_lock:
        history = [dict(sample) for sample in _gpu_history]
    if not history:
        return {"current": None, "history": [], "age_seconds": None}
    latest = {key: value for key, value in history[-1].items() if key != "t"}
    current = time.time() if now is None else now
    age = round(current - history[-1].get("t", current), 1)
    return {"current": latest, "history": history, "age_seconds": age}



# Reuse unchanged run summaries across half-second polls. The rollout panels
# each scan the complete history, so rebuilding only after a file changes keeps
# monitoring work off the CPU while training is generating or updating.
_RUN_CACHE_LIMIT = 16
_run_cache: OrderedDict[str, tuple[tuple[Any, ...], dict[str, Any]]] = OrderedDict()
_run_cache_lock = RLock()


def _file_signature(path: Path) -> tuple[Any, ...]:
    try:
        stat = path.stat()
        return str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size
    except OSError:
        return str(path), None


def _cached_collect_run(
    arm_hint: str, run_dir: Path, now: float, live_arms: set[str] | None,
) -> dict[str, Any]:
    watched = (META_NAME, EVAL_NAME, METRICS_NAME, ROLLOUTS_NAME, CONTROL_NAME, CONTROL_EVENTS_NAME)
    signature = tuple(_file_signature(run_dir / name) for name in watched)
    signature += tuple(_file_signature(path) for _, path in _question_text_files())
    key = str(run_dir.resolve())
    with _run_cache_lock:
        cached = _run_cache.get(key)
        if cached is None or cached[0] != signature:
            snapshot = _collect_run(arm_hint, run_dir, now, live_arms)
            _run_cache[key] = signature, snapshot
        else:
            snapshot = cached[1]
        _run_cache.move_to_end(key)
        while len(_run_cache) > _RUN_CACHE_LIMIT:
            _run_cache.popitem(last=False)
        result = deepcopy(snapshot)

    # Process state and ages can change while every run file stays unchanged.
    result["metrics_age_seconds"] = _metrics_age(run_dir / METRICS_NAME, now)
    alive = None if live_arms is None else result["arm"] in live_arms
    result["status"] = _status(_read_eval(run_dir), run_dir / METRICS_NAME, now, alive)
    started = _parse_timestamp(result["started"])
    result["elapsed_seconds"] = round(now - started, 1) if started is not None else None
    return result


def collect_dashboard(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Collect the dashboard data for every run under the given paths."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.now(timezone.utc).timestamp()
    live_arms = _process_arms()
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for arm, run_dir in _resolve_paths(paths):
        if run_dir is None:
            runs.append(_waiting_run(arm))
            continue
        key = str(run_dir)
        if key in seen:
            continue
        seen.add(key)
        runs.append(_cached_collect_run(arm, run_dir, now, live_arms))
    return {
        "generated_at": generated_at,
        "runs": runs,
        "gpu": _gpu_state(now),
    }


def _embed_json(data: Mapping[str, Any]) -> str:
    """JSON safe to put inside a script element."""
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


WEB_DIR = Path(__file__).with_name("web")
PAGE_NAME = "index.html"
DATA_MARKER = "__RLVR_INITIAL_DATA__"

_page_template: str | None = None

_FALLBACK_PAGE = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>RLVR</title></head>
<body style="background:#08080a;color:#9a9aa3;font:12px ui-monospace,monospace;padding:24px">
the dashboard bundle is missing. run `npm install && npm run build` in dashboard/
</body>
</html>
"""


def _page_template_text() -> str:
    """The built page, read once per process."""
    global _page_template
    if _page_template is None:
        try:
            _page_template = (WEB_DIR / PAGE_NAME).read_text(encoding="utf-8")
        except OSError:
            _page_template = _FALLBACK_PAGE
    return _page_template


def render_html(data: Mapping[str, Any]) -> str:
    """The built dashboard page with the first payload inlined.

    User strings ride in JSON and are rendered by React as text, so nothing
    from a run folder becomes HTML markup here. The JS bundle was inlined by
    the build, so the page makes no network requests beyond `/data.json`.
    """
    return _page_template_text().replace(DATA_MARKER, _embed_json(data), 1)


class _DashboardHandler(BaseHTTPRequestHandler):
    """One HTTP handler. Paths are bound on a subclass per server."""

    paths: list[str] = []
    server_version = "rlvr-dashboard"

    def do_GET(self) -> None:
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _route(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            body = render_html(collect_dashboard(self.paths)).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/data.json":
            sample_gpu()
            body = (json.dumps(collect_dashboard(self.paths), separators=(",", ":")) + "\n").encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def serve_dashboard(
    paths: Iterable[str | Path], *, host: str = "127.0.0.1", port: int = 8081
) -> None:
    """Serve the dashboard until Ctrl+C."""
    handler = type(
        "_BoundHandler", (_DashboardHandler,), {"paths": [str(path) for path in paths]}
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    print(f"dashboard: http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
