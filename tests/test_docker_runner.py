"""Tests for the Docker test runner: isolation, timeout, per-test results, classes."""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from rlvr.data import MBPPQuestion, load_dataset, prompt_for_question
from rlvr.docker_runner import (
    DEFAULT_IMAGE,
    DEFAULT_TIMEOUT_SECONDS,
    ERROR_CLASSES,
    DockerTestRunner,
    docker_command,
)
from rlvr.experiment import Prompt

PROMPT_ID = "mbpp-1"


def question(*tests: str, setup: str = "") -> MBPPQuestion:
    return MBPPQuestion(
        task_id=1,
        text="Return one.",
        test_list=tuple(tests),
        test_setup_code=setup,
        source_split="test",
    )


def prompt() -> Prompt:
    return Prompt(prompt_id=PROMPT_ID, problem="Return one.")


@pytest.fixture(scope="session")
def docker_ready() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not installed")
    try:
        image = subprocess.run(
            ["docker", "image", "inspect", DEFAULT_IMAGE],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        pytest.skip(f"docker is not usable: {error}")
    if image.returncode != 0:
        pytest.skip(f"docker image {DEFAULT_IMAGE} is not available locally")
    probe = subprocess.run(
        ["docker", "run", "--rm", DEFAULT_IMAGE, "python", "-c", "print(1)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(f"docker cannot start {DEFAULT_IMAGE}: {probe.stderr.strip()}")


@pytest.fixture
def runner(docker_ready) -> DockerTestRunner:
    return DockerTestRunner([question("assert f() == 1")])


def run(runner: DockerTestRunner, code: str):
    return runner.run(prompt(), code)


def test_error_class_names_are_stable() -> None:
    assert ERROR_CLASSES == (
        "pass",
        "wrong_answer",
        "syntax_error",
        "runtime_error",
        "timeout",
        "no_tests",
        "runner_error",
    )


def test_defaults_match_the_spec() -> None:
    assert DEFAULT_IMAGE == "python:3.12-slim"
    assert DEFAULT_TIMEOUT_SECONDS == 5.0


def test_docker_command_has_the_isolation_flags(tmp_path) -> None:
    command = docker_command(
        image=DEFAULT_IMAGE,
        task_dir=tmp_path,
        container_name="rlvr-probe",
    )
    text = " ".join(command)
    assert command[:2] == ["docker", "run"]
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--user" in command and command[command.index("--user") + 1] == "65534:65534"
    assert "--read-only" in command
    assert "--privileged" not in command
    volume = command[command.index("--volume") + 1]
    assert volume.endswith(":/task:ro")
    assert str(tmp_path) in volume


def test_docker_command_passes_no_parent_environment(tmp_path) -> None:
    command = docker_command(
        image=DEFAULT_IMAGE,
        task_dir=tmp_path,
        container_name="rlvr-probe",
    )
    env_values = [
        command[index + 1]
        for index, part in enumerate(command)
        if part in ("--env", "-e")
    ]
    assert env_values == ["PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1"]
    assert not any("SECRET" in part for part in command)


def test_passing_code_is_classified_pass(runner: DockerTestRunner) -> None:
    outcome = run(runner, "def f():\n    return 1\n")
    assert outcome.tests_passed == 1
    assert outcome.tests_total == 1
    assert outcome.error_class == "pass"
    assert [result.passed for result in outcome.results] == [True]
    assert [result.error_class for result in outcome.results] == ["pass"]


def test_wrong_answer_is_classified_wrong_answer(runner: DockerTestRunner) -> None:
    outcome = run(runner, "def f():\n    return 2\n")
    assert outcome.tests_passed == 0
    assert outcome.error_class == "wrong_answer"
    assert outcome.results[0].error_class == "wrong_answer"


def test_syntax_error_is_classified(runner: DockerTestRunner) -> None:
    outcome = run(runner, "def f(:\n    return 1\n")
    assert outcome.tests_passed == 0
    assert outcome.tests_total == 1
    assert outcome.error_class == "syntax_error"


def test_runtime_error_is_classified(runner: DockerTestRunner) -> None:
    outcome = run(runner, "def f():\n    return 1 / 0\n")
    assert outcome.tests_passed == 0
    assert outcome.error_class == "runtime_error"
    assert outcome.results[0].error_class == "runtime_error"


def test_module_level_failure_is_a_runtime_error(runner: DockerTestRunner) -> None:
    outcome = run(runner, "raise ValueError('boom')\n")
    assert outcome.tests_passed == 0
    assert outcome.error_class == "runtime_error"


def test_hung_code_is_killed_and_classified_timeout(docker_ready) -> None:
    runner = DockerTestRunner([question("assert f() == 1")], timeout=2.0)
    started = time.monotonic()
    outcome = runner.run(prompt(), "def f():\n    while True:\n        pass\n")
    elapsed = time.monotonic() - started
    assert outcome.tests_passed == 0
    assert outcome.error_class == "timeout"
    assert elapsed < 15.0
    assert all(result.error_class == "timeout" for result in outcome.results)


def test_each_test_runs_separately_and_reports_partial_passes(docker_ready) -> None:
    runner = DockerTestRunner(
        [
            question(
                "assert f() == 1",
                "assert f() == 2",
                "assert f() == 1",
            )
        ]
    )
    outcome = runner.run(prompt(), "def f():\n    return 1\n")
    assert outcome.tests_total == 3
    assert outcome.tests_passed == 2
    assert outcome.error_class == "wrong_answer"
    assert [result.index for result in outcome.results] == [0, 1, 2]
    assert [result.passed for result in outcome.results] == [True, False, True]
    assert [result.error_class for result in outcome.results] == [
        "pass",
        "wrong_answer",
        "pass",
    ]


def test_test_setup_code_runs_before_the_tests(docker_ready) -> None:
    runner = DockerTestRunner(
        [question("assert f() == helper()", setup="def helper():\n    return 3\n")]
    )
    outcome = runner.run(prompt(), "def f():\n    return 3\n")
    assert outcome.tests_passed == 1
    assert outcome.error_class == "pass"


def test_secret_in_parent_environment_does_not_reach_the_container(
    docker_ready, monkeypatch
) -> None:
    monkeypatch.setenv("RLVR_TEST_SECRET", "s3cr3t-value")
    runner = DockerTestRunner(
        [
            question(
                "assert leak() == 'clean'",
            )
        ]
    )
    code = (
        "import os\n"
        "def leak():\n"
        "    return os.environ.get('RLVR_TEST_SECRET', 'clean')\n"
    )
    outcome = runner.run(prompt(), code)
    assert outcome.error_class == "pass"
    assert outcome.tests_passed == 1


def test_container_writes_are_blocked(docker_ready) -> None:
    runner = DockerTestRunner([question("assert f() == 1")])
    code = (
        "def f():\n"
        "    with open('/task/evil.txt', 'w') as handle:\n"
        "        handle.write('x')\n"
        "    return 1\n"
    )
    outcome = runner.run(prompt(), code)
    assert outcome.error_class == "runtime_error"


def test_the_container_has_no_network(docker_ready) -> None:
    runner = DockerTestRunner([question("assert not reachable()")])
    code = (
        "import socket\n"
        "def reachable():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 80), timeout=1)\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
    )
    outcome = runner.run(prompt(), code)
    assert outcome.error_class == "pass"
    assert outcome.tests_passed == 1


def test_a_task_with_no_tests_is_classified_no_tests(docker_ready) -> None:
    runner = DockerTestRunner([question()])
    outcome = runner.run(prompt(), "def f():\n    return 1\n")
    assert outcome.tests_total == 0
    assert outcome.tests_passed == 0
    assert outcome.error_class == "no_tests"


def test_a_broken_docker_binary_is_classified_runner_error(docker_ready) -> None:
    runner = DockerTestRunner(
        [question("assert f() == 1")],
        docker_bin="/nonexistent/docker",
    )
    outcome = runner.run(prompt(), "def f():\n    return 1\n")
    assert outcome.error_class == "runner_error"
    assert outcome.tests_passed == 0


def test_a_real_mbpp_row_runs_every_test(docker_ready) -> None:
    question_602 = next(
        item for item in load_dataset() if item.task_id == 602
    )
    runner = DockerTestRunner([question_602])
    code = (
        "def first_repeated_char(text):\n"
        "    seen = set()\n"
        "    for character in text:\n"
        "        if character in seen:\n"
        "            return character\n"
        "        seen.add(character)\n"
        "    return 'None'\n"
    )
    outcome = runner.run(prompt_for_question(question_602), code)
    assert outcome.tests_total == len(question_602.test_list)
    assert outcome.tests_passed == outcome.tests_total
    assert outcome.error_class == "pass"


def test_unknown_prompt_id_is_an_error(runner: DockerTestRunner) -> None:
    with pytest.raises(KeyError):
        runner.run(Prompt(prompt_id="mbpp-999999", problem="x"), "def f():\n    return 1\n")
