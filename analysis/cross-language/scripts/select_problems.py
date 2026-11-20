"""Build the 20-problem benchmark manifest shared by all four languages.

Python comes from the original HumanEval file. The other three come from the
MultiPL-E parquet files, converted to jsonl by the download script. Problems
are matched by HumanEval number, then sampled once with a fixed seed.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA = Path("/home/opencode/Work/qwen-finetune-analysis/data")
OUT = Path("/home/opencode/Work/qwen-finetune-analysis/data/manifest.json")
SEED = 2026
COUNT = 20


def number_from_name(name: str) -> int:
    match = re.search(r"HumanEval[_/](\d+)", name)
    if not match:
        raise ValueError(f"no HumanEval number in {name!r}")
    return int(match.group(1))


def load_python() -> dict[int, dict]:
    rows = {}
    with (DATA / "humaneval-py.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[number_from_name(row["task_id"])] = row
    return rows


def load_translated(language: str) -> dict[int, dict]:
    rows = {}
    with (DATA / f"humaneval-{language}.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[number_from_name(row["name"])] = row
    return rows


def main() -> None:
    python_rows = load_python()
    translated = {lang: load_translated(lang) for lang in ("js", "java", "cpp")}
    common = sorted(
        set(python_rows)
        & set(translated["js"])
        & set(translated["java"])
        & set(translated["cpp"])
    )
    chosen = sorted(random.Random(SEED).sample(common, COUNT))
    manifest = []
    for number in chosen:
        manifest.append(
            {
                "number": number,
                "python": python_rows[number],
                "js": translated["js"][number],
                "java": translated["java"][number],
                "cpp": translated["cpp"][number],
            }
        )
    OUT.write_text(json.dumps({"seed": SEED, "count": COUNT, "problems": manifest}, indent=1))
    print(f"intersection {len(common)}, sampled {len(manifest)}")
    for entry in manifest:
        print(entry["number"], entry["python"]["entry_point"], "|", entry["java"]["name"])


if __name__ == "__main__":
    main()
