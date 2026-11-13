"""A live progress dashboard served over a local HTTP server.

The dashboard reads the same run folders the summary reads, but never fails on
a run that is mid-write. It serves two resources:

    GET /          a self-contained page with Tailwind and Chart.js
    GET /data.json the JSON the page polls every five seconds

The page is a monitoring tool, not a report. One section per arm, one KPI strip
per section, and a grid of charts. Distributions come from the tail of
`rollouts.jsonl`, so a long run does not re-read the whole file on every poll.
GPU samples come from `nvidia-smi`, one sample per data request, and the last
120 samples are kept in memory for the GPU chart.

The server binds to 127.0.0.1 and stops on Ctrl+C. Nothing here reads secrets,
environment variables, or the network beyond the two CDN scripts the page
fetches in the browser.
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

__all__ = ["collect_dashboard", "render_html", "sample_gpu", "serve_dashboard"]

METRICS_NAME = "metrics.jsonl"
EVAL_NAME = "eval.json"
META_NAME = "meta.json"
ROLLOUTS_NAME = "rollouts.jsonl"

#: A metrics file not touched in this many seconds counts as stopped, unless a
#: final evaluation is already on disk.
STALE_SECONDS = 60.0

#: How many bytes of rollouts.jsonl to read from the tail for distributions.
ROLLOUT_TAIL_BYTES = 4 * 1024 * 1024

#: Number of bins for the reward histogram.
REWARD_BINS = 20

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

SERIES_FIELDS: tuple[str, ...] = (
    "mean_reward",
    "pass_rate",
    "mean_entropy",
    "mean_completion_tokens",
    "truncation_rate",
    "cache_hits",
)

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


def _target_steps(meta: Mapping[str, Any]) -> int | None:
    """Target steps derivable from meta.json, or None when it is not recorded."""
    config = meta.get("config")
    if not isinstance(config, dict):
        return None
    training = config.get("training")
    data = config.get("data")
    if isinstance(training, dict) and isinstance(data, dict):
        epochs = _positive_int(training.get("epochs"))
        train_size = _positive_int(data.get("train_size"))
        if epochs is not None and train_size is not None:
            target = epochs * train_size
            budget = config.get("budget")
            if isinstance(budget, dict):
                max_steps = _positive_int(budget.get("max_steps"))
                if max_steps is not None:
                    target = min(target, max_steps)
            return target
    fake = config.get("fake")
    if isinstance(fake, dict):
        steps = _positive_int(fake.get("steps"))
        if steps is not None:
            return steps
    return None


def _steps_done(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    step = _integer(rows[-1].get("step"))
    return step if step is not None and step > 0 else len(rows)


def _latest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {name: None for name in LATEST_FIELDS}
    last = rows[-1]
    result: dict[str, Any] = {}
    for name in SERIES_FIELDS:
        result[name] = _integer(last.get(name)) if name == "cache_hits" else _number(last.get(name))
    result["rollouts"] = _integer(last.get("rollouts"))
    return result


def _series(rows: list[dict[str, Any]]) -> dict[str, Any]:
    series: dict[str, Any] = {"steps": []}
    for name in SERIES_FIELDS:
        series[name] = []
    for row in rows:
        series["steps"].append(_integer(row.get("step")))
        for name in SERIES_FIELDS:
            value = _integer(row.get(name)) if name == "cache_hits" else _number(row.get(name))
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
        "elapsed_seconds": None,
        "status": "waiting",
        "steps_done": 0,
        "target_steps": None,
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
        "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
        "status": _status(eval_payload, run_dir / METRICS_NAME, now),
        "steps_done": _steps_done(rows),
        "target_steps": _target_steps(meta),
        "latest": _latest(rows),
        "series": _series(rows),
        "pass_at_1": _pass_at_1_series(eval_payload),
        "evaluations": _evaluations(eval_payload),
        "distributions": _distributions(run_dir),
    }


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
        "utilization": _number(parts[0]),
        "memory_used": _number(parts[1]),
        "memory_total": _number(parts[2]),
        "temperature": _number(parts[3]),
    }


def sample_gpu() -> None:
    """Sample nvidia-smi once and keep the sample in the rolling history."""
    sample = _gpu_read()
    if sample is not None:
        sample["t"] = time.time()
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


_STYLE = """
html { color-scheme: dark; }
canvas { display: block; }
.tnum { font-variant-numeric: tabular-nums; }
"""

_SCRIPT = """\
(function () {
  "use strict";
  var ACCENT = "#fbbf24";
  var MUTED = "#a1a1aa";
  var DATA = JSON.parse(document.getElementById("initial-data").textContent);
  var charts = [];
  var timer = null;

  function esc(s) {
    if (s === null || s === undefined) { return ""; }
    return String(s).replace(/[&<>"']/g, function (c) {
      if (c === "&") { return "&amp;"; }
      if (c === "<") { return "&lt;"; }
      if (c === ">") { return "&gt;"; }
      if (c === '"') { return "&quot;"; }
      return "&#39;";
    });
  }
  function fmt(v, d) {
    if (v === null || v === undefined) { return "-"; }
    if (d === undefined) { d = 4; }
    return Number(v).toFixed(d);
  }
  function fmtElapsed(sec) {
    if (sec === null || sec === undefined) { return "-"; }
    var total = Math.floor(sec);
    var h = Math.floor(total / 3600);
    var m = Math.floor((total % 3600) / 60);
    var s = total % 60;
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    if (h) { return h + "h " + pad(m) + "m " + pad(s) + "s"; }
    if (m) { return m + "m " + pad(s) + "s"; }
    return s + "s";
  }
  function xy(steps, values) {
    var out = [];
    for (var i = 0; i < steps.length; i++) {
      if (steps[i] === null || steps[i] === undefined) { continue; }
      out.push({ x: steps[i], y: (values[i] === undefined ? null : values[i]) });
    }
    return out;
  }
  function pill(status) {
    var map = {
      running: "text-amber-300 border-amber-500/40",
      finished: "text-emerald-300 border-emerald-500/40",
      stopped: "text-red-300 border-red-500/40",
      waiting: "text-zinc-400 border-zinc-600"
    };
    var cls = map[status] || "text-zinc-400 border-zinc-600";
    return '<span class="inline-block px-2 py-0.5 border rounded text-[10px] uppercase tracking-wider ' +
      cls + '">' + esc(status) + '</span>';
  }
  function kpi(label, value) {
    return '<div class="bg-zinc-900 px-3 py-2">' +
      '<div class="text-[10px] uppercase tracking-wider text-zinc-500">' + esc(label) + '</div>' +
      '<div class="text-lg font-medium text-zinc-100 tnum">' + value + '</div>' +
      '</div>';
  }
  function chartCard(id, label) {
    return '<div class="border border-zinc-800 bg-zinc-900/40 p-3">' +
      '<div class="text-[11px] uppercase tracking-wider text-zinc-500 mb-2">' + esc(label) + '</div>' +
      '<div class="relative h-40"><canvas id="' + id + '"></canvas></div></div>';
  }
  function runHeader(run) {
    var target = (run.target_steps === null || run.target_steps === undefined) ? "" : " / " + run.target_steps;
    return '<div class="flex items-center justify-between flex-wrap gap-2 border-b border-zinc-800 pb-2 mb-3">' +
      '<div class="flex items-baseline gap-2 min-w-0">' +
        '<h2 class="text-sm font-semibold text-zinc-100">' + esc(run.arm) + '</h2>' +
        '<span class="text-xs text-zinc-500 font-mono truncate">' + esc(run.run_name || "") + '</span>' +
      '</div>' +
      '<div class="flex items-center gap-3 text-xs text-zinc-500">' +
        pill(run.status) +
        '<span class="tnum">elapsed ' + fmtElapsed(run.elapsed_seconds) + '</span>' +
        '<span class="tnum">steps ' + (run.steps_done || 0) + target + '</span>' +
      '</div>' +
      '</div>';
  }
  function evaluationTable(evals) {
    var html = '<div class="mt-3 border border-zinc-800">';
    html += '<div class="px-3 py-2 text-[11px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">evaluations</div>';
    html += '<table class="w-full text-xs">';
    html += '<thead><tr class="text-left text-zinc-500">' +
      '<th class="px-3 py-1.5 font-normal">step</th>' +
      '<th class="px-3 py-1.5 font-normal">kind</th>' +
      '<th class="px-3 py-1.5 font-normal text-right">count</th>' +
      '<th class="px-3 py-1.5 font-normal text-right">pass@1</th></tr></thead><tbody>';
    if (!evals.length) {
      html += '<tr><td colspan="4" class="px-3 py-2 text-zinc-500">no evaluation recorded</td></tr>';
    } else {
      for (var e = 0; e < evals.length; e++) {
        var it = evals[e];
        html += '<tr class="border-t border-zinc-800/60">' +
          '<td class="px-3 py-1.5 tnum">' + (it.step === null || it.step === undefined ? "-" : it.step) + '</td>' +
          '<td class="px-3 py-1.5">' + esc(it.kind || "") + '</td>' +
          '<td class="px-3 py-1.5 text-right tnum">' + (it.count === null || it.count === undefined ? "-" : it.count) + '</td>' +
          '<td class="px-3 py-1.5 text-right tnum">' + fmt(it.pass_at_1) + '</td></tr>';
      }
    }
    html += '</tbody></table></div>';
    return html;
  }
  function renderRun(run, i) {
    var p = "r" + i;
    var status = run.status || "waiting";
    var latest = run.latest || {};
    var evals = run.evaluations || [];
    var lastEval = evals.length ? evals[evals.length - 1] : null;

    var html = '<section class="mb-6">' + runHeader(run);
    if (status === "waiting") {
      html += '<div class="py-8 text-center text-sm text-zinc-500">waiting for a run</div></section>';
      return html;
    }

    var cacheRate = "-";
    if (latest.cache_hits !== null && latest.cache_hits !== undefined && latest.rollouts) {
      cacheRate = (latest.cache_hits / latest.rollouts).toFixed(3);
    }

    html += '<div class="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-8 gap-px bg-zinc-800 border border-zinc-800 mb-3">';
    html += kpi("steps", (run.steps_done || 0) + (run.target_steps === null || run.target_steps === undefined ? "" : " / " + run.target_steps));
    html += kpi("mean reward", fmt(latest.mean_reward));
    html += kpi("pass rate", fmt(latest.pass_rate));
    html += kpi("pass@1", lastEval ? fmt(lastEval.pass_at_1) : "-");
    html += kpi("entropy", fmt(latest.mean_entropy));
    html += kpi("comp len", fmt(latest.mean_completion_tokens, 1));
    html += kpi("truncation", fmt(latest.truncation_rate));
    html += kpi("cache hit rate", cacheRate);
    html += '</div>';

    html += '<div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">';
    html += chartCard(p + "-reward", "reward per step");
    html += chartCard(p + "-pass", "pass rate per step");
    html += chartCard(p + "-passat1", "pass@1");
    html += chartCard(p + "-entropy", "entropy per step");
    html += chartCard(p + "-length", "completion tokens per step");
    html += chartCard(p + "-trunc", "truncation rate per step");
    html += chartCard(p + "-rewardhist", "reward distribution");
    html += chartCard(p + "-testshist", "tests passed distribution");
    html += '</div>';

    html += evaluationTable(evals);
    html += '</section>';
    return html;
  }
  function renderGpu(gpu) {
    var cur = gpu.current;
    var html = '<section class="mb-6">';
    html += '<h2 class="text-sm font-semibold text-zinc-100 border-b border-zinc-800 pb-2 mb-3">system</h2>';
    html += '<div class="grid grid-cols-2 sm:grid-cols-4 gap-px bg-zinc-800 border border-zinc-800 mb-3">';
    html += kpi("gpu util", cur && cur.utilization !== null && cur.utilization !== undefined ? cur.utilization.toFixed(0) + "%" : "-");
    html += kpi("gpu memory", cur && cur.memory_used !== null && cur.memory_used !== undefined ? (cur.memory_used + " MiB" + (cur.memory_total !== null && cur.memory_total !== undefined ? " / " + cur.memory_total + " MiB" : "")) : "-");
    html += kpi("gpu temp", cur && cur.temperature !== null && cur.temperature !== undefined ? cur.temperature.toFixed(0) + " C" : "-");
    html += kpi("samples", String((gpu.history || []).length));
    html += '</div>';
    if ((gpu.history || []).length) {
      html += '<div class="grid grid-cols-1 gap-3">' + chartCard("gpu-chart", "gpu utilization and memory") + '</div>';
    }
    html += '</section>';
    return html;
  }
  function render(data) {
    charts.forEach(function (c) { try { c.destroy(); } catch (e) {} });
    charts = [];
    var app = document.getElementById("app");
    var runs = data.runs || [];
    var html = "";
    if (!runs.length) {
      html = '<div class="py-16 text-center text-sm text-zinc-500">waiting for a run</div>';
    } else {
      if (data.gpu && data.gpu.current) { html += renderGpu(data.gpu); }
      for (var i = 0; i < runs.length; i++) { html += renderRun(runs[i], i); }
    }
    app.innerHTML = html;
    if (!runs.length) {
      document.getElementById("footer").innerHTML = "source -  |  refresh 5s";
      return;
    }
    if (data.gpu && data.gpu.current && data.gpu.history && data.gpu.history.length) {
      mountGpuChart(data.gpu);
    }
    for (var j = 0; j < runs.length; j++) {
      if (runs[j].status !== "waiting") { mountRunCharts(runs[j], j); }
    }
    var sources = runs.map(function (r) { return esc(r.run_dir || ""); }).join("   ");
    document.getElementById("footer").innerHTML = "source " + sources + "  |  refresh 5s";
  }
  function lineOpts() {
    return {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#18181b", borderColor: "#3f3f46", borderWidth: 1 } },
      scales: {
        x: { type: "linear", grid: { color: "#27272a" }, ticks: { color: "#71717a", maxTicksLimit: 8, maxRotation: 0 } },
        y: { grid: { color: "#27272a" }, ticks: { color: "#71717a" } }
      }
    };
  }
  function mount(id, cfg) {
    var canvas = document.getElementById(id);
    if (!canvas) { return; }
    charts.push(new Chart(canvas, cfg));
  }
  function lineDs(color, data) {
    return { data: data, borderColor: color, backgroundColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0, spanGaps: false };
  }
  function bar(id, labels, counts, color) {
    mount(id, {
      type: "bar",
      data: { labels: labels, datasets: [{ data: counts, backgroundColor: color, borderWidth: 0 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { backgroundColor: "#18181b", borderColor: "#3f3f46", borderWidth: 1 } },
        scales: {
          x: { grid: { display: false }, ticks: { color: "#71717a", maxRotation: 0, autoSkip: true, maxTicksLimit: 10 } },
          y: { grid: { color: "#27272a" }, ticks: { color: "#71717a", precision: 0 }, beginAtZero: true }
        }
      }
    });
  }
  function mountRunCharts(run, i) {
    var p = "r" + i;
    var s = run.series || {};
    var steps = s.steps || [];
    var d = function (color, values) { return lineDs(color, xy(steps, values || [])); };
    mount(p + "-reward", { type: "line", data: { datasets: [d(ACCENT, s.mean_reward)] }, options: lineOpts() });
    mount(p + "-pass", { type: "line", data: { datasets: [d(ACCENT, s.pass_rate)] }, options: lineOpts() });
    mount(p + "-entropy", { type: "line", data: { datasets: [d(ACCENT, s.mean_entropy)] }, options: lineOpts() });
    mount(p + "-length", { type: "line", data: { datasets: [d(ACCENT, s.mean_completion_tokens)] }, options: lineOpts() });
    mount(p + "-trunc", { type: "line", data: { datasets: [d(ACCENT, s.truncation_rate)] }, options: lineOpts() });

    var pa = run.pass_at_1 || {};
    var cp = pa.checkpoint || { steps: [], values: [] };
    var fn = pa.final || { steps: [], values: [] };
    mount(p + "-passat1", {
      type: "line",
      data: { datasets: [
        { label: "checkpoint", data: xy(cp.steps || [], cp.values || []), borderColor: MUTED, backgroundColor: MUTED, borderWidth: 1.5, pointRadius: 2, pointBackgroundColor: MUTED, tension: 0, spanGaps: false },
        { label: "final", data: xy(fn.steps || [], fn.values || []), borderColor: ACCENT, backgroundColor: ACCENT, borderWidth: 1.5, pointRadius: 2, pointBackgroundColor: ACCENT, tension: 0, spanGaps: false }
      ] },
      options: Object.assign(lineOpts(), { plugins: { legend: { display: true, labels: { color: "#a1a1aa", boxWidth: 10, boxHeight: 10 } } } })
    });

    var dist = run.distributions || {};
    if (dist.reward && dist.reward.bins) {
      bar(p + "-rewardhist", dist.reward.bins.map(function (b) { return b.toFixed(2); }), dist.reward.counts, "#3f3f46");
    }
    if (dist.tests_passed && dist.tests_passed.bins) {
      bar(p + "-testshist", dist.tests_passed.bins.map(function (b) { return String(b); }), dist.tests_passed.counts, "#3f3f46");
    }
  }
  function mountGpuChart(gpu) {
    var hist = gpu.history || [];
    var t = hist.map(function (h) { return h.t; });
    var util = hist.map(function (h) { return h.utilization; });
    var mem = hist.map(function (h) { return h.memory_used; });
    mount("gpu-chart", {
      type: "line",
      data: { datasets: [
        { label: "utilization %", data: xy(t, util), borderColor: ACCENT, backgroundColor: ACCENT, borderWidth: 1.5, pointRadius: 0, tension: 0, spanGaps: false, yAxisID: "y" },
        { label: "memory MiB", data: xy(t, mem), borderColor: MUTED, backgroundColor: MUTED, borderWidth: 1.5, pointRadius: 0, tension: 0, spanGaps: false, yAxisID: "y1" }
      ] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: true, labels: { color: "#a1a1aa", boxWidth: 10, boxHeight: 10 } }, tooltip: { backgroundColor: "#18181b", borderColor: "#3f3f46", borderWidth: 1 } },
        scales: {
          x: { type: "linear", grid: { color: "#27272a" }, ticks: { color: "#71717a", maxTicksLimit: 8, maxRotation: 0 } },
          y: { position: "left", grid: { color: "#27272a" }, ticks: { color: "#71717a" } },
          y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { color: "#71717a" } }
        }
      }
    });
  }
  function refresh() {
    fetch("/data.json").then(function (r) { return r.json(); }).then(function (data) {
      document.getElementById("updated").textContent = data.generated_at || "-";
      render(data);
    }).catch(function () {});
  }
  function start() { stop(); timer = setInterval(refresh, 5000); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }
  document.getElementById("autorefresh").addEventListener("change", function (e) {
    if (e.target.checked) { refresh(); start(); } else { stop(); }
  });

  Chart.defaults.color = MUTED;
  Chart.defaults.borderColor = "#27272a";
  Chart.defaults.font.family = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  Chart.defaults.font.size = 11;

  render(DATA);
  start();
})();
"""


def render_html(data: Mapping[str, Any]) -> str:
    """The dashboard page shell. User strings ride in JSON and are rendered by
    the browser, so nothing from a run folder becomes HTML markup here."""
    generated = _string(data.get("generated_at")) or "-"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>RLVR dashboard</title>",
        '<script src="https://cdn.tailwindcss.com"></script>',
        "<script>",
        "tailwind.config = {",
        "  theme: {",
        "    extend: {",
        "      colors: { accent: '#fbbf24', muted: '#a1a1aa' },",
        "    },",
        "  },",
        "};",
        "</script>",
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>',
        f"<style>\n{_STYLE}</style>",
        "</head>",
        '<body class="bg-zinc-950 text-zinc-300 antialiased">',
        '<header class="border-b border-zinc-800">',
        '<div class="max-w-[1440px] mx-auto px-4 py-3 flex items-center justify-between gap-4">',
        '<div class="flex items-baseline gap-3">',
        '<span class="text-sm font-semibold text-zinc-100">RLVR</span>',
        '<span class="text-xs text-zinc-500 uppercase tracking-wider">training monitor</span>',
        "</div>",
        '<div class="flex items-center gap-4 text-xs text-zinc-500">',
        '<span class="tnum">updated <span id="updated">' + generated + "</span></span>",
        '<label class="flex items-center gap-2">',
        '<input type="checkbox" id="autorefresh" checked class="accent-amber-400">',
        "<span>auto-refresh</span>",
        "</label>",
        "</div>",
        "</div>",
        "</header>",
        '<main id="app" class="max-w-[1440px] mx-auto px-4 py-4"></main>',
        '<footer id="footer" class="max-w-[1440px] mx-auto px-4 pb-6 text-xs text-zinc-600"></footer>',
        '<script type="application/json" id="initial-data">' + _embed_json(data) + "</script>",
        "<script>" + _SCRIPT + "</script>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


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
        sample_gpu()
        if path in ("/", "/index.html"):
            body = render_html(collect_dashboard(self.paths)).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/data.json":
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
