"""Compile and run each generated answer against its shipped tests.

Runs on the GPU box. Python uses the original HumanEval test block plus its
`check(entry_point)` call. JavaScript, Java and C++ use the MultiPL-E `tests`
string exactly as shipped, appended to the prompt and completion.

Writes one JSON per model and language with the outcome of every question.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

JAVATUPLES = "/home/Ubuntu/qwen-bench/lib/javatuples-1.2.jar"
NODE = os.environ.get("NODE_BIN", "node")
COMPILE_TIMEOUT = 40
RUN_TIMEOUT = 10


def run(command: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, (result.stdout + result.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"


def grade_python(row: dict, work: Path) -> tuple[str, str]:
    source = (
        row["prompt"]
        + row["completion"]
        + "\n\n"
        + row["test"]
        + f"\ncheck({row['entry_point']})\n"
    )
    path = work / "main.py"
    path.write_text(source)
    return run([sys.executable, str(path)], work, RUN_TIMEOUT)


def grade_js(row: dict, work: Path) -> tuple[str, str]:
    source = row["prompt"] + row["completion"] + "\n" + row["tests"]
    path = work / "main.js"
    path.write_text(source)
    return run([NODE, str(path)], work, RUN_TIMEOUT)


def drop_repeated_closing_brace(completion: str, tests: str) -> str:
    """Remove one trailing brace line when the shipped tests also close it.

    The Java and C++ test strings begin with the closing brace of the function
    under test. A completion that closes the function itself would otherwise
    produce one brace too many. This trims only that single trailing line.
    """
    trimmed = completion.rstrip()
    if not tests.lstrip().startswith("}") or not trimmed.endswith("}"):
        return completion
    cut = trimmed.rfind("\n")
    if cut != -1 and trimmed[cut + 1 :].strip() == "}":
        return completion[:cut]
    return completion


def grade_java(row: dict, work: Path) -> tuple[str, str]:
    match = re.search(r"class\s+(\w+)", row["prompt"])
    if not match:
        return 2, "no class name in prompt"
    name = match.group(1)
    completion = drop_repeated_closing_brace(row["completion"], row["tests"])
    source = row["prompt"] + completion + "\n" + row["tests"]
    (work / f"{name}.java").write_text(source)
    code, output = run(
        [
            "javac",
            "-cp",
            JAVATUPLES,
            "-d",
            str(work),
            str(work / f"{name}.java"),
        ],
        work,
        COMPILE_TIMEOUT,
    )
    if code != 0:
        return 2, "compile:\n" + output
    return run(
        ["java", "-ea", "-cp", f"{work}:{JAVATUPLES}", name], work, RUN_TIMEOUT
    )


def grade_cpp(row: dict, work: Path) -> tuple[str, str]:
    completion = drop_repeated_closing_brace(row["completion"], row["tests"])
    source = row["prompt"] + completion + "\n" + row["tests"]
    path = work / "main.cpp"
    path.write_text(source)
    code, output = run(
        ["g++", "-std=c++17", "-O1", "-o", str(work / "main"), str(path)],
        work,
        COMPILE_TIMEOUT,
    )
    if code != 0:
        return 2, "compile:\n" + output
    return run([str(work / "main")], work, RUN_TIMEOUT)


GRADERS = {
    "python": grade_python,
    "js": grade_js,
    "java": grade_java,
    "cpp": grade_cpp,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", required=True, choices=sorted(GRADERS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.completions).read_text())
    manifest = json.loads(Path(args.manifest).read_text())["problems"]
    by_number = {entry["number"]: entry for entry in manifest}
    language = args.language
    grader = GRADERS[language]

    outcomes = []
    for row in payload["rows"]:
        number = row["number"]
        base_row = dict(by_number[number][language])
        base_row["completion"] = row["completion"]
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            code, output = grader(base_row, work)
        if code == 0:
            status, stage = "pass", "run"
        elif code == 2:
            status, stage = "fail", "compile"
        elif code == 124:
            status, stage = "fail", "timeout"
        else:
            status, stage = "fail", "run"
        outcomes.append(
            {
                "number": number,
                "name": row["name"],
                "status": status,
                "stage": stage,
                "returncode": code,
                "output_tail": output.strip()[-600:],
            }
        )
        print(f"{payload['model']} {language} {number} {status} {stage}", flush=True)

    passed = sum(1 for row in outcomes if row["status"] == "pass")
    result = {
        "model": payload["model"],
        "language": language,
        "problems": len(outcomes),
        "passed": passed,
        "outcomes": outcomes,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}: {passed}/{len(outcomes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
