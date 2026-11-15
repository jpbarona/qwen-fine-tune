"""A live progress dashboard served over a local HTTP server.

The dashboard reads the same run folders the summary reads, but never fails on
a run that is mid-write. It serves two resources:

    GET /          one HTML page with the React bundle inlined
    GET /data.json the JSON the page polls every five seconds

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
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import urlsplit

from .memory import MEMORY_METRIC_FIELDS
from .stages import BANDS

__all__ = ["collect_dashboard", "render_html", "sample_gpu", "serve_dashboard"]

METRICS_NAME = "metrics.jsonl"
EVAL_NAME = "eval.json"
META_NAME = "meta.json"
ROLLOUTS_NAME = "rollouts.jsonl"

#: A metrics file not touched in this many seconds counts as stopped, unless a
#: final evaluation is already on disk. One step is a vLLM batch plus a test run
#: per rollout, which took about a minute on the 3090 and longer on wide batches,
#: so a one minute window reported finished runs as stopped.
STALE_SECONDS = 900.0

#: How many bytes of rollouts.jsonl to read from the tail for distributions.
ROLLOUT_TAIL_BYTES = 4 * 1024 * 1024

#: Number of bins for the reward histogram.
REWARD_BINS = 20

#: A GPU sample younger than this many seconds is reused instead of sampled
#: again, so a page that polls fast does not spawn nvidia-smi every request.
GPU_SAMPLE_SECONDS = 2.0

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

SERIES_FIELDS: tuple[str, ...] = (
    "mean_reward",
    "pass_rate",
    "mean_entropy",
    "mean_completion_tokens",
    "truncation_rate",
    "cache_hits",
    "prompts",
    "sequences",
) + MEMORY_METRIC_FIELDS

#: Fields that are whole counts and are read as integers, not floats.
COUNT_SERIES_FIELDS: tuple[str, ...] = ("cache_hits", "prompts", "sequences")

LATEST_FIELDS: tuple[str, ...] = SERIES_FIELDS + ("rollouts",)

#: The last 120 GPU samples, one per data request.
_gpu_history: deque[dict[str, Any]] = deque(maxlen=120)


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_meta(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / META_NAME)


def _read_eval(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / EVAL_NAME)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    """Read metrics.jsonl, skipping a line that is still being written."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_tail_lines(path: Path, max_bytes: int) -> list[str]:
    """The last bytes of a file, as lines. The first partial line is dropped."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []
    try:
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]
    return lines


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


def _real_target_steps(meta: Mapping[str, Any], config: Mapping[str, Any]) -> int | None:
    """Epochs times the train size, divided into batches of prompts.

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
    train_size = _positive_int(data.get("train_size"))
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


def _steps_done(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    step = _integer(rows[-1].get("step"))
    return step if step is not None and step > 0 else len(rows)


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
    counts = [0] * REWARD_BINS
    for reward in rewards:
        index = min(int(reward * REWARD_BINS), REWARD_BINS - 1)
        counts[index] += 1
    bins = [index / REWARD_BINS for index in range(REWARD_BINS)]
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
    """Full pass rate per curriculum band, from the rollout tail.

    A rollout counts as passed only when every test passed, matching the
    training pool rule. Bands appear in the fixed order first, then any band
    name the run folder carries that is not one of the four.
    """
    counts: dict[str, int] = {}
    passed: dict[str, int] = {}
    for record in _rollout_records(run_dir / ROLLOUTS_NAME):
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


def _stage_progress(run_dir: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """How much of the current curriculum band has been solved, per step.

    A prompt counts as solved in a step only when every rollout of that prompt
    in that step passed, which is the training pool rule. The total for a band
    is its starting training pool from `meta.json`, so the percentage is stable
    even though the pool shrinks.

    The series keeps one point per training step, and carries the band with it,
    because a run that moves to a new band restarts at zero.
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

    by_step: dict[int, dict[str, dict[str, list[bool]]]] = {}
    for record in _rollout_records(run_dir / ROLLOUTS_NAME):
        step = _integer(record.get("step"))
        band = _string(record.get("stage"))
        prompt_id = _string(record.get("prompt_id"))
        if step is None or band is None or prompt_id is None:
            continue
        passed = _integer(record.get("tests_passed"))
        total = _integer(record.get("tests_total"))
        full = passed is not None and total is not None and total > 0 and passed == total
        step_band = by_step.setdefault(step, {}).setdefault(band, {})
        step_band.setdefault(prompt_id, []).append(full)

    solved: dict[str, set[str]] = {}
    points: list[dict[str, Any]] = []
    for step in sorted(by_step):
        for band, prompts in by_step[step].items():
            for prompt_id, outcomes in prompts.items():
                if outcomes and all(outcomes):
                    solved.setdefault(band, set()).add(prompt_id)
            total = totals.get(band)
            if not total:
                continue
            points.append(
                {
                    "step": step,
                    "band": band,
                    "solved": len(solved.get(band, ())),
                    "total": total,
                    "percent": len(solved.get(band, ())) / total,
                }
            )

    latest = points[-1] if points else None
    return {
        "band": latest["band"] if latest else None,
        "solved": latest["solved"] if latest else 0,
        "total": latest["total"] if latest else 0,
        "percent": latest["percent"] if latest else None,
        "points": points,
    }


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


def _status(eval_payload: Mapping[str, Any], metrics_path: Path, now: float) -> str:
    if any(point.get("kind") == "final" for point in _raw_evaluations(eval_payload)):
        return "finished"
    if not metrics_path.is_file():
        return "waiting"
    try:
        mtime = metrics_path.stat().st_mtime
    except OSError:
        return "stopped"
    if now - mtime <= STALE_SECONDS:
        return "running"
    return "stopped"


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
        "stage": None,
        "bands": [],
        "stage_progress": {"band": None, "solved": 0, "total": 0, "percent": None, "points": []},
        "latest": {name: None for name in LATEST_FIELDS},
        "series": {"steps": []} | {name: [] for name in SERIES_FIELDS},
        "pass_at_1": {"checkpoint": {"steps": [], "values": []}, "final": {"steps": [], "values": []}},
        "evaluations": [],
        "distributions": {"reward": {"bins": [], "counts": []}, "tests_passed": {"bins": [], "counts": []}},
    }


def _collect_run(arm_hint: str, run_dir: Path, now: float) -> dict[str, Any]:
    meta = _read_meta(run_dir)
    eval_payload = _read_eval(run_dir)
    rows = _read_metrics(run_dir / METRICS_NAME)

    arm = _string(meta.get("arm")) or _string(eval_payload.get("arm")) or arm_hint or run_dir.name
    timestamp = _string(meta.get("timestamp")) or _string(eval_payload.get("timestamp")) or run_dir.name
    started = _parse_timestamp(timestamp)
    elapsed = now - started if started is not None else None

    return {
        "arm": arm,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "started": timestamp,
        "started_utc": _started_label(timestamp),
        "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
        "status": _status(eval_payload, run_dir / METRICS_NAME, now),
        "metrics_age_seconds": _metrics_age(run_dir / METRICS_NAME, now),
        "steps_done": _steps_done(rows),
        "target_steps": _target_steps(meta),
        "stage": _string(rows[-1].get("stage")) if rows else None,
        "bands": _band_pass_rates(run_dir),
        "stage_progress": _stage_progress(run_dir, meta),
        "latest": _latest(rows),
        "series": _series(rows),
        "pass_at_1": _pass_at_1_series(eval_payload),
        "evaluations": _evaluations(eval_payload),
        "distributions": _distributions(run_dir),
    }


def _gpu_number(value: str) -> float | None:
    """An nvidia-smi CSV field as a number, or None when it is not one.

    nvidia-smi prints text tokens such as N/A, [N/A], [Not Supported] and
    [Unknown Error] for a field it cannot report, so anything that does not
    parse as a finite number becomes None.
    """
    try:
        return _number(float(value.strip()))
    except ValueError:
        return None


def _gpu_read() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 4:
        return None
    return {
        "utilization": _gpu_number(parts[0]),
        "memory_used": _gpu_number(parts[1]),
        "memory_total": _gpu_number(parts[2]),
        "temperature": _gpu_number(parts[3]),
    }


def sample_gpu(now: float | None = None) -> None:
    """Sample nvidia-smi unless the last sample is still fresh.

    The sample is kept in the rolling history. A sample younger than
    `GPU_SAMPLE_SECONDS` is reused, so the 404 and HTML paths do not run
    nvidia-smi and a fast poll does not either.
    """
    current = time.time() if now is None else now
    if _gpu_history:
        age = current - _gpu_history[-1].get("t", 0.0)
        if age < GPU_SAMPLE_SECONDS:
            return
    sample = _gpu_read()
    if sample is not None:
        sample["t"] = current
        _gpu_history.append(sample)


def _gpu_state() -> dict[str, Any]:
    if not _gpu_history:
        return {"current": None, "history": []}
    latest = {key: value for key, value in _gpu_history[-1].items() if key != "t"}
    return {"current": latest, "history": list(_gpu_history)}


def collect_dashboard(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Collect the dashboard data for every run under the given paths."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.now(timezone.utc).timestamp()
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
        runs.append(_collect_run(arm, run_dir, now))
    return {"generated_at": generated_at, "runs": runs, "gpu": _gpu_state()}


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
        self.send_header("Cache-Control", "no-store")
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
