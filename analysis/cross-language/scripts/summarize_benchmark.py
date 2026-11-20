"""Combine the eight grading files into one summary with per-question outcomes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/opencode/Work/qwen-finetune-analysis")
RESULTS = ROOT / "results"
LANGUAGES = ("python", "js", "java", "cpp")


def main() -> None:
    table = {}
    for model in ("base", "adapter"):
        table[model] = {}
        for language in LANGUAGES:
            payload = json.loads((RESULTS / f"{model}-{language}.json").read_text())
            table[model][language] = payload

    summary = {
        "problems": 20,
        "attempts_per_problem": 1,
        "decoding": "greedy, raw completion prompt, max 512 new tokens",
        "languages": {},
        "per_question": [],
        "totals": {},
    }

    for language in LANGUAGES:
        base = table["base"][language]
        adapter = table["adapter"][language]
        base_map = {row["number"]: row for row in base["outcomes"]}
        adapter_map = {row["number"]: row for row in adapter["outcomes"]}
        numbers = sorted(base_map)
        gained = [n for n in numbers if adapter_map[n]["status"] == "pass" and base_map[n]["status"] != "pass"]
        lost = [n for n in numbers if base_map[n]["status"] == "pass" and adapter_map[n]["status"] != "pass"]
        summary["languages"][language] = {
            "base_passed": base["passed"],
            "adapter_passed": adapter["passed"],
            "problems": len(numbers),
            "gained": gained,
            "lost": lost,
        }
        for number in numbers:
            summary["per_question"].append(
                {
                    "number": number,
                    "language": language,
                    "base": base_map[number]["status"],
                    "adapter": adapter_map[number]["status"],
                    "base_stage": base_map[number]["stage"],
                    "adapter_stage": adapter_map[number]["stage"],
                }
            )

    summary["totals"] = {
        "base": sum(v["base_passed"] for v in summary["languages"].values()),
        "adapter": sum(v["adapter_passed"] for v in summary["languages"].values()),
        "out_of": 20 * len(LANGUAGES),
    }
    (RESULTS / "benchmark-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["languages"], indent=1))
    print(json.dumps(summary["totals"]))


if __name__ == "__main__":
    main()
