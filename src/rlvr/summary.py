"""Read run folders, build the summary, and render the static report.

This module only reads files a run already wrote. It never touches a model,
judge, Docker, or the network.

A run folder holds `metrics.jsonl` (one row per step) and `eval.json` (the final
pass@1 plus every checkpoint evaluation). `summary` joins the two on the step
number. Steps without an evaluation show a blank pass@1.

The HTML is one static file. Styles are inline in a single `style` element, and
there is no script, font, image, or stylesheet fetched from anywhere.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .reward import ARMS

__all__ = [
    "RunReport",
    "StepMetric",
    "SummaryError",
    "format_summary",
    "load_report",
    "load_reports",
    "render_html",
    "resolve_run_dirs",
    "write_report",
]

METRICS_NAME = "metrics.jsonl"
EVAL_NAME = "eval.json"
META_NAME = "meta.json"

#: Step fields read from metrics.jsonl, in validation order. Entropy is read
#: separately because it is present only on a step that measured it.
_STEP_FIELDS: tuple[tuple[str, str], ...] = (
    ("mean_reward", "mean reward"),
    ("pass_rate", "pass rate"),
    ("mean_completion_tokens", "mean completion tokens"),
)

#: Tables printed by `format_summary`, in order.
_METRIC_TABLES: tuple[tuple[str, str], ...] = (
    ("mean_reward", "reward per step"),
    ("pass_at_1", "pass@1 per step"),
    ("mean_entropy", "entropy per step"),
    ("mean_completion_tokens", "length per step"),
)


class SummaryError(ValueError):
    """A run folder is missing, unreadable, or does not hold the expected data."""


@dataclass(frozen=True)
class StepMetric:
    """One metrics row, joined with any checkpoint pass@1 at that step."""

    step: int
    mean_reward: float
    pass_rate: float
    mean_entropy: float | None
    mean_completion_tokens: float
    rollouts: int
    pass_at_1: float | None


@dataclass(frozen=True)
class RunReport:
    """One run folder, as the summary and report need it."""

    arm: str
    run_dir: Path
    timestamp: str
    steps: tuple[StepMetric, ...]
    final_pass_at_1: float | None
    final_count: int | None
    checkpoint_subset: Mapping[str, Any] | None

    @property
    def rollouts(self) -> int:
        return sum(step.rollouts for step in self.steps)

    def mean(self, field: str) -> float | None:
        """The mean of a step field over the steps that carry a value.

        Entropy is measured only every `entropy_every_steps` training steps, so a step without
        a value is skipped rather than counted as zero. With no measured step
        the result is None.
        """
        values = []
        for step in self.steps:
            value = getattr(step, field)
            if value is not None:
                values.append(value)
        if not values:
            return None
        return sum(values) / len(values)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"{path}: cannot read JSON: {error}") from error


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SummaryError(f"{path}: cannot read: {error}") from error

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SummaryError(f"{path}: line {number}: not valid JSON: {error}") from error
        if not isinstance(row, dict):
            raise SummaryError(f"{path}: line {number}: row must be an object")
        rows.append(row)
    return rows


def _number(row: Mapping[str, Any], key: str, source: str) -> float:
    if key not in row:
        raise SummaryError(f"{source}: missing field {key!r}")
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(
            f"{source}: field {key!r} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise SummaryError(f"{source}: field {key!r} must be finite, got {value!r}")
    return number


def _optional_number(row: Mapping[str, Any], key: str, source: str) -> float | None:
    """A number from `row`, or None when the key is absent or null.

    Entropy is written only on a measurement step. A missing key means the step
    did not measure it, which is not an error.
    """
    if key not in row or row[key] is None:
        return None
    return _number(row, key, source)


def _step_number(row: Mapping[str, Any], source: str) -> int:
    if "step" not in row:
        raise SummaryError(f"{source}: missing field 'step'")
    value = row["step"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SummaryError(f"{source}: field 'step' must be a positive integer, got {value!r}")
    return value


def _checkpoint_pass_at_1(payload: Mapping[str, Any], source: str) -> dict[int, float]:
    """Map step to the last checkpoint or final evaluation pass@1 at that step.

    A stage evaluation carries its own band name and is shown separately. It
    never replaces a checkpoint value for the per-step pass@1 line.
    """
    points = payload.get("evaluations")
    if points is None:
        points = []
    if not isinstance(points, list):
        raise SummaryError(f"{source}: 'evaluations' must be a list")

    result: dict[int, float] = {}
    for point in points:
        if not isinstance(point, dict):
            raise SummaryError(f"{source}: each evaluation must be an object")
        if point.get("kind") == "stage":
            continue
        step = _step_number(point, source)
        result[step] = _number(point, "pass_at_1", source)
    return result


def _optional_final(payload: Mapping[str, Any], source: str) -> tuple[float | None, int | None]:
    if "pass_at_1" not in payload and "count" not in payload:
        return None, None
    return _number(payload, "pass_at_1", source), int(_number(payload, "count", source))


def _last_checkpoint_subset(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """The subset from the last checkpoint evaluation, when the top level omits it."""
    points = payload.get("evaluations")
    if not isinstance(points, list):
        return None
    found: dict[str, Any] | None = None
    for point in points:
        if not isinstance(point, dict):
            continue
        subset = point.get("subset")
        if isinstance(subset, dict):
            found = subset
    return found


def load_report(run_dir: str | Path) -> RunReport:
    """Read one run folder. Raises SummaryError with the file in the message."""
    folder = Path(run_dir)
    metrics_path = folder / METRICS_NAME
    if not metrics_path.is_file():
        raise SummaryError(f"{folder}: not a run folder, missing {METRICS_NAME}")

    meta: dict[str, Any] = {}
    if (folder / META_NAME).is_file():
        loaded = _load_json(folder / META_NAME)
        if isinstance(loaded, dict):
            meta = loaded

    eval_payload: dict[str, Any] = {}
    if (folder / EVAL_NAME).is_file():
        loaded = _load_json(folder / EVAL_NAME)
        if not isinstance(loaded, dict):
            raise SummaryError(f"{folder / EVAL_NAME}: expected a JSON object")
        eval_payload = loaded

    source = str(metrics_path)
    checkpoint = _checkpoint_pass_at_1(eval_payload, str(folder / EVAL_NAME))
    rows = _jsonl(metrics_path)
    steps: list[StepMetric] = []
    for row in rows:
        step = _step_number(row, source)
        values = {key: _number(row, key, source) for key, _ in _STEP_FIELDS}
        rollouts_value = row.get("rollouts", 0)
        if isinstance(rollouts_value, bool) or not isinstance(rollouts_value, int):
            raise SummaryError(f"{source}: field 'rollouts' must be an integer")
        steps.append(
            StepMetric(
                step=step,
                mean_reward=values["mean_reward"],
                pass_rate=values["pass_rate"],
                mean_entropy=_optional_number(row, "mean_entropy", source),
                mean_completion_tokens=values["mean_completion_tokens"],
                rollouts=rollouts_value,
                pass_at_1=checkpoint.get(step),
            )
        )

    final_pass_at_1, final_count = _optional_final(eval_payload, source)
    arm = meta.get("arm") or eval_payload.get("arm") or folder.parent.name
    if not isinstance(arm, str):
        raise SummaryError(f"{folder}: arm must be a string")
    timestamp = meta.get("timestamp") or eval_payload.get("timestamp") or folder.name
    if not isinstance(timestamp, str):
        raise SummaryError(f"{folder}: timestamp must be a string")
    subset = eval_payload.get("subset")
    if subset is None:
        subset = _last_checkpoint_subset(eval_payload)
    if subset is not None and not isinstance(subset, dict):
        raise SummaryError(f"{folder / EVAL_NAME}: 'subset' must be an object or null")

    return RunReport(
        arm=arm,
        run_dir=folder,
        timestamp=timestamp,
        steps=tuple(steps),
        final_pass_at_1=final_pass_at_1,
        final_count=final_count,
        checkpoint_subset=subset,
    )


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / METRICS_NAME).is_file()


def _expand(path: Path, *, required: bool) -> list[Path]:
    if _is_run_dir(path):
        return [path]
    if path.is_dir():
        found: list[Path] = []
        for child in sorted(path.iterdir()):
            if child.is_dir():
                found.extend(_expand(child, required=False))
        if found:
            return found
    if required:
        raise SummaryError(f"{path}: not a run folder")
    return []


def resolve_run_dirs(paths: Iterable[str | Path]) -> list[Path]:
    """Turn run folders, arm folders, and runs roots into run folders.

    A runs root expands to every `<arm>/<timestamp>` folder below it. Repeated
    paths are kept once, in input order.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        for path in _expand(Path(raw), required=True):
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                found.append(path)
    return found


def load_reports(paths: Iterable[str | Path]) -> tuple[RunReport, ...]:
    """Load every run under the given paths, ordered by arm then timestamp."""
    reports = [load_report(path) for path in resolve_run_dirs(paths)]

    def order(report: RunReport) -> tuple[int, str, str]:
        try:
            arm_index = ARMS.index(report.arm)
        except ValueError:
            arm_index = len(ARMS)
        return arm_index, report.timestamp, str(report.run_dir)

    return tuple(sorted(reports, key=order))


def _labels(reports: Sequence[RunReport]) -> list[str]:
    """Column labels. A repeated arm also shows its timestamp."""
    totals: dict[str, int] = {}
    for report in reports:
        totals[report.arm] = totals.get(report.arm, 0) + 1

    labels: list[str] = []
    used: dict[str, int] = {}
    for report in reports:
        label = report.arm if totals[report.arm] == 1 else f"{report.arm}@{report.timestamp}"
        count = used.get(label, 0) + 1
        used[label] = count
        if count > 1:
            label = f"{label}#{count}"
        labels.append(label)
    return labels


def _value(step: StepMetric, field: str) -> str:
    value = getattr(step, field)
    if value is None:
        return "-"
    if field == "mean_completion_tokens":
        return f"{value:.1f}"
    return f"{value:.4f}"


def format_summary(reports: Sequence[RunReport]) -> str:
    """The plain text summary: one table per metric, arms side by side."""
    if not reports:
        return "no runs\n"

    labels = _labels(reports)
    lines: list[str] = []
    for report in reports:
        lines.append(f"run {report.arm} {report.run_dir}")
        if report.final_pass_at_1 is not None:
            lines.append(
                f"  final pass@1 {report.final_pass_at_1:.4f} "
                f"over {report.final_count} questions"
            )
    lines.append("")

    lookups = [{step.step: step for step in report.steps} for report in reports]
    all_steps = sorted({step.step for report in reports for step in report.steps})
    step_width = max(4, max((len(str(step)) for step in all_steps), default=4))
    label_widths = [max(len(label), 8) for label in labels]

    for field, title in _METRIC_TABLES:
        lines.append(title)
        header = "step".ljust(step_width)
        header += "".join("  " + label.rjust(width) for label, width in zip(labels, label_widths))
        lines.append(header)
        for step_number in all_steps:
            row = str(step_number).ljust(step_width)
            for lookup, width in zip(lookups, label_widths):
                step = lookup.get(step_number)
                text = "-" if step is None else _value(step, field)
                row += "  " + text.rjust(width)
            lines.append(row)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _subset_text(subset: Mapping[str, Any] | None) -> str:
    if not subset:
        return "No checkpoint evaluation was recorded."
    method = subset.get("method", "unknown")
    seed = subset.get("seed")
    selected = subset.get("selected")
    pool = subset.get("pool_size")
    if method == "seeded_sample":
        return (
            f"Checkpoint subset: seeded sample, seed {seed}, "
            f"{selected} of {pool} questions."
        )
    return f"Checkpoint subset: {method}, seed {seed}, {selected} of {pool} questions."


def _format_optional(value: float | None) -> str:
    return "not available" if value is None else f"{value:.4f}"


_STYLE = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 62rem; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a;
  background: #ffffff; line-height: 1.5; }
h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; }
h2 { font-size: 1.15rem; font-weight: 600; margin-top: 2rem;
  border-bottom: 1px solid #dddddd; padding-bottom: 0.25rem; }
p { margin: 0.5rem 0; }
.meta { color: #555555; font-size: 0.9rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem;
  margin: 0.75rem 0 1.5rem 0; }
th, td { text-align: right; padding: 0.25rem 0.5rem;
  border-bottom: 1px solid #e5e5e5; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 600; background: #f7f7f7; }
""".strip()


def render_html(
    reports: Sequence[RunReport], *, generated_at: str | None = None
) -> str:
    """A static, self-contained HTML report with one section per arm."""
    generated = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    escape = html.escape

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>RLVR reward study</title>",
        f"<style>\n{_STYLE}\n</style>",
        "</head>",
        "<body>",
        "<h1>RLVR reward study</h1>",
        f'<p class="meta">Generated {escape(generated)}. pass@1 only.</p>',
    ]

    parts.append("<h2>Arms</h2>")
    parts.append("<table>")
    parts.append(
        "<thead><tr><th>arm</th><th>steps</th><th>rollouts</th><th>mean reward</th>"
        "<th>final pass@1</th><th>mean entropy</th><th>mean length</th></tr></thead>"
    )
    parts.append("<tbody>")
    for label, report in zip(_labels(reports), reports):
        mean_reward = report.mean("mean_reward")
        mean_entropy = report.mean("mean_entropy")
        mean_length = report.mean("mean_completion_tokens")
        parts.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{len(report.steps)}</td>"
            f"<td>{report.rollouts}</td>"
            f"<td>{_format_optional(mean_reward)}</td>"
            f"<td>{_format_optional(report.final_pass_at_1)}</td>"
            f"<td>{_format_optional(mean_entropy)}</td>"
            f"<td>{'-' if mean_length is None else f'{mean_length:.1f}'}</td>"
            "</tr>"
        )
    parts.append("</tbody>")
    parts.append("</table>")

    for label, report in zip(_labels(reports), reports):
        parts.append('<section class="arm">')
        parts.append(f"<h2>{escape(label)}</h2>")
        parts.append(
            f'<p class="meta">run: {escape(str(report.run_dir))}<br>'
            f"started: {escape(report.timestamp)}</p>"
        )
        if report.final_pass_at_1 is None:
            parts.append("<p>Final pass@1: not available.</p>")
        else:
            parts.append(
                f"<p>Final pass@1: {report.final_pass_at_1:.4f} "
                f"over {report.final_count} questions.</p>"
            )
        parts.append(f"<p>{escape(_subset_text(report.checkpoint_subset))}</p>")
        parts.append("<table>")
        parts.append(
            "<thead><tr><th>step</th><th>reward</th><th>pass@1</th>"
            "<th>entropy</th><th>length</th></tr></thead>"
        )
        parts.append("<tbody>")
        for step in report.steps:
            parts.append(
                "<tr>"
                f"<td>{step.step}</td>"
                f"<td>{step.mean_reward:.4f}</td>"
                f"<td>{_value(step, 'pass_at_1')}</td>"
                f"<td>{_value(step, 'mean_entropy')}</td>"
                f"<td>{step.mean_completion_tokens:.1f}</td>"
                "</tr>"
            )
        parts.append("</tbody>")
        parts.append("</table>")
        parts.append("</section>")

    parts.extend(["</body>", "</html>", ""])
    return "\n".join(parts)


def write_report(
    path: str | Path,
    reports: Sequence[RunReport],
    *,
    generated_at: str | None = None,
) -> Path:
    """Write the HTML report and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(reports, generated_at=generated_at), encoding="utf-8")
    return target
