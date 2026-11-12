"""Run candidate code against MBPP task tests in a locked-down Docker container.

Every run uses the same fixed flags:

    --network none          no network
    --user 65534:65534      non-root
    --read-only             read-only root filesystem
    --tmpfs /tmp            the only writable place, not used by the harness
    --volume <task>:/task:ro  read-only task directory
    --env ...               only the two variables below, never the parent env
    --rm                    remove the container afterwards

The 5 second timeout is enforced from outside with `subprocess.run`. A container
that outlives it is killed by name, so a hung candidate cannot keep running.

The harness runs the candidate module once, then runs each test string
separately in the same namespace. Assertion failures are `wrong_answer`. Any
other exception is `runtime_error`. A candidate that does not compile is
`syntax_error`. A task with no tests is `no_tests`. A container that fails to
start or produce a result is `runner_error`. A full pass is `pass`.

Error classes are stable and must not be renamed:

    pass, wrong_answer, syntax_error, runtime_error, timeout, no_tests,
    runner_error

The candidate can print anything. The runner reads only the last line that
starts with the marker below.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

from .data import MBPPQuestion, prompt_id_for
from .experiment import Outcome, Prompt, TestResult

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_TIMEOUT_SECONDS",
    "ERROR_CLASSES",
    "DockerTestRunner",
    "docker_command",
]

#: Small official image with Python 3.12. Override with a local equivalent.
DEFAULT_IMAGE = "python:3.12-slim"

#: Wall-clock limit for one candidate, including container startup.
DEFAULT_TIMEOUT_SECONDS = 5.0

#: Stable error class names. `pass` means every test passed.
ERROR_CLASSES: tuple[str, ...] = (
    "pass",
    "wrong_answer",
    "syntax_error",
    "runtime_error",
    "timeout",
    "no_tests",
    "runner_error",
)

#: The non-root user inside python:3.12-slim.
CONTAINER_USER = "65534:65534"

#: Prefix of the JSON result line printed by the harness.
RESULT_MARKER = "RLVR_RESULT:"

#: The harness, copied into the read-only task directory. It contains no secret.
_HARNESS_SOURCE = '''\
import json
import sys

MARKER = "RLVR_RESULT:"


def detail(error):
    text = type(error).__name__ + ": " + str(error)
    return text.replace("\\n", " ")[:300]


def emit(payload):
    sys.stdout.write(MARKER + json.dumps(payload) + "\\n")
    sys.stdout.flush()


with open("/task/task.json", encoding="utf-8") as handle:
    task = json.load(handle)

namespace = {"__name__": "candidate"}

try:
    compiled = compile(task["code"], "candidate.py", "exec")
except SyntaxError as error:
    emit({"error_class": "syntax_error", "results": [], "detail": detail(error)})
    raise SystemExit(0)

try:
    exec(compiled, namespace)
except BaseException as error:
    emit({"error_class": "runtime_error", "results": [], "detail": detail(error)})
    raise SystemExit(0)

setup = task.get("test_setup_code") or ""
if setup:
    try:
        exec(compile(setup, "test_setup.py", "exec"), namespace)
    except BaseException as error:
        emit({"error_class": "runtime_error", "results": [], "detail": detail(error)})
        raise SystemExit(0)

tests = task.get("test_list") or []
if not tests:
    emit({"error_class": "no_tests", "results": [], "detail": ""})
    raise SystemExit(0)

results = []
failures = []
for index, test in enumerate(tests):
    try:
        exec(compile(test, "test_%d.py" % index, "exec"), namespace)
    except AssertionError as error:
        results.append(
            {
                "index": index,
                "passed": False,
                "error_class": "wrong_answer",
                "detail": detail(error),
            }
        )
        failures.append("wrong_answer")
    except BaseException as error:
        results.append(
            {
                "index": index,
                "passed": False,
                "error_class": "runtime_error",
                "detail": detail(error),
            }
        )
        failures.append("runtime_error")
    else:
        results.append(
            {"index": index, "passed": True, "error_class": "pass", "detail": ""}
        )

if all(result["passed"] for result in results):
    error_class = "pass"
elif "runtime_error" in failures:
    error_class = "runtime_error"
else:
    error_class = "wrong_answer"

emit({"error_class": error_class, "results": results, "detail": ""})
'''


def docker_command(
    *,
    image: str = DEFAULT_IMAGE,
    task_dir: str | Path,
    container_name: str,
    docker_bin: str = "docker",
) -> list[str]:
    """The exact `docker run` command. Kept public so tests can check the flags."""
    return [
        docker_bin,
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--user",
        CONTAINER_USER,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=16m",
        "--volume",
        f"{Path(task_dir)}:/task:ro",
        "--workdir",
        "/task",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        image,
        "python",
        "-I",
        "-B",
        "/task/harness.py",
    ]


def _detail(text: str, limit: int = 300) -> str:
    return text.replace("\n", " ").strip()[:limit]


class DockerTestRunner:
    """The concrete TestRunner seam. One container per candidate."""

    # The class name starts with "Test", so tell pytest this is not a test class.
    __test__ = False

    def __init__(
        self,
        questions: Iterable[MBPPQuestion],
        *,
        image: str = DEFAULT_IMAGE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        docker_bin: str = "docker",
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self.image = image
        self.timeout = float(timeout)
        self.docker_bin = docker_bin
        self._questions: dict[str, MBPPQuestion] = {}
        for question in questions:
            key = prompt_id_for(question)
            if key in self._questions:
                raise ValueError(f"duplicate question for prompt id {key!r}")
            self._questions[key] = question

    def run(self, prompt: Prompt, code: str) -> Outcome:
        """Run one candidate. Never raises for bad candidate code."""
        try:
            question = self._questions[prompt.prompt_id]
        except KeyError:
            raise KeyError(
                f"no task tests for prompt id {prompt.prompt_id!r}"
            ) from None

        tests_total = len(question.test_list)
        task = {
            "code": code,
            "test_list": list(question.test_list),
            "test_setup_code": question.test_setup_code,
        }

        container_name = f"rlvr-{uuid.uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory(prefix="rlvr-task-") as directory:
            task_dir = Path(directory)
            (task_dir / "task.json").write_text(
                json.dumps(task, ensure_ascii=False), encoding="utf-8"
            )
            (task_dir / "harness.py").write_text(_HARNESS_SOURCE, encoding="utf-8")
            os.chmod(task_dir, 0o755)
            os.chmod(task_dir / "task.json", 0o644)
            os.chmod(task_dir / "harness.py", 0o644)

            command = docker_command(
                image=self.image,
                task_dir=task_dir,
                container_name=container_name,
                docker_bin=self.docker_bin,
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                self._kill(container_name)
                return self._timeout_outcome(tests_total)
            except OSError:
                return self._runner_error(tests_total)

        return self._parse(completed, tests_total)

    def _kill(self, container_name: str) -> None:
        for args in (["kill", container_name], ["rm", "-f", container_name]):
            try:
                subprocess.run(
                    [self.docker_bin, *args],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass

    @staticmethod
    def _timeout_outcome(tests_total: int) -> Outcome:
        results = tuple(
            TestResult(index=index, passed=False, error_class="timeout")
            for index in range(tests_total)
        )
        return Outcome(
            tests_passed=0,
            tests_total=tests_total,
            error_class="timeout",
            results=results,
        )

    @staticmethod
    def _runner_error(tests_total: int) -> Outcome:
        return Outcome(
            tests_passed=0,
            tests_total=tests_total,
            error_class="runner_error",
            results=(),
        )

    def _parse(self, completed: subprocess.CompletedProcess[str], tests_total: int) -> Outcome:
        payload = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(RESULT_MARKER):
                try:
                    payload = json.loads(line[len(RESULT_MARKER) :])
                except json.JSONDecodeError:
                    payload = None
                break

        if not isinstance(payload, dict) or "error_class" not in payload:
            return self._runner_error(tests_total)

        error_class = payload["error_class"]
        if not isinstance(error_class, str) or error_class not in ERROR_CLASSES:
            return self._runner_error(tests_total)

        raw_results = payload.get("results")
        results: list[TestResult] = []
        if isinstance(raw_results, list):
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                passed = item.get("passed")
                if isinstance(index, bool) or not isinstance(index, int):
                    continue
                if not isinstance(passed, bool):
                    continue
                item_class = item.get("error_class", "pass" if passed else "wrong_answer")
                if not isinstance(item_class, str):
                    item_class = "pass" if passed else "wrong_answer"
                item_detail = item.get("detail", "")
                if not isinstance(item_detail, str):
                    item_detail = ""
                results.append(
                    TestResult(
                        index=index,
                        passed=passed,
                        error_class=item_class,
                        detail=_detail(item_detail),
                    )
                )

        passed_count = sum(1 for result in results if result.passed)
        return Outcome(
            tests_passed=passed_count,
            tests_total=tests_total,
            error_class=error_class,
            results=tuple(results),
        )
