"""Fetch the four benchmark files and convert them to jsonl.

Python comes from the OpenAI HumanEval repository as the original jsonl.gz.
MultiPL-E has no Python config, so the other three come from the parquet files
in nuprl/MultiPL-E. Run this from the analysis folder with pyarrow installed.
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

DATA = Path("/home/opencode/Work/qwen-finetune-analysis/data")
BASE = "https://huggingface.co/datasets/nuprl/MultiPL-E/resolve/main"
LANGUAGES = ("js", "java", "cpp")


def fetch(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    raw = DATA / "HumanEval.jsonl.gz"
    fetch(
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        raw,
    )
    (DATA / "humaneval-py.jsonl").write_bytes(gzip.decompress(raw.read_bytes()))

    import pyarrow.parquet as pq

    for language in LANGUAGES:
        parquet = DATA / f"humaneval-{language}.parquet"
        if not parquet.exists():
            fetch(f"{BASE}/humaneval-{language}/test-00000-of-00001.parquet", parquet)
        rows = pq.read_table(parquet).to_pylist()
        with (DATA / f"humaneval-{language}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(language, len(rows))
    print("python", sum(1 for _ in (DATA / "humaneval-py.jsonl").open()))


if __name__ == "__main__":
    main()
