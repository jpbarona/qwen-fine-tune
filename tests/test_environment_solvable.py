"""Cold check that the environment a run uses can actually be solved.

The point is to catch a broken prompt or a broken harness before a GPU run,
not afterwards. Two checks:

- every synthetic reference solution passes its own tests inside the real
  container harness, not just under `exec` in the test process,
- a candidate that uses the wrong function name is reported as a runtime
  error, which is the failure shape a missing name produces.

Docker is required. The whole file is skipped when it is not available.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from rlvr.data import expected_call_arity, expected_function_name, prompt_id_for
from rlvr.docker_runner import DEFAULT_IMAGE, DockerTestRunner
from rlvr.experiment import Prompt
from rlvr.synthetic import load_synthetic_questions


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


def test_every_synthetic_reference_solution_passes_in_the_real_harness(
    docker_ready,
) -> None:
    questions = [question.as_question() for question in load_synthetic_questions()]
    runner = DockerTestRunner(questions)
    failures = []
    for question, source in zip(questions, load_synthetic_questions()):
        prompt = Prompt(prompt_id=prompt_id_for(question), problem=question.text)
        outcome = runner.run(prompt, source.code)
        if outcome.error_class != "pass":
            failures.append(
                (
                    question.task_id,
                    outcome.error_class,
                    [item.detail for item in outcome.results][:1],
                )
            )
    assert failures == []


def test_a_wrong_function_name_is_a_runtime_error(docker_ready) -> None:
    question = load_synthetic_questions()[0].as_question()
    runner = DockerTestRunner([question])
    prompt = Prompt(prompt_id=prompt_id_for(question), problem=question.text)
    name = expected_function_name(question)
    wrong = "def not_the_name(*args, **kwargs):\n    return None\n"
    outcome = runner.run(prompt, wrong)
    assert name != "not_the_name"
    assert outcome.error_class == "runtime_error"
    assert any("NameError" in item.detail for item in outcome.results)


def test_a_replaced_test_list_is_the_one_that_scores(docker_ready) -> None:
    """The control file's `tests` edit must reach the container, not just memory."""
    question = load_synthetic_questions()[0].as_question()
    runner = DockerTestRunner([question])
    prompt = Prompt(prompt_id=prompt_id_for(question), problem=question.text)

    source = load_synthetic_questions()[0].code
    assert runner.run(prompt, source).error_class == "pass"

    name = expected_function_name(question)
    arity = expected_call_arity(question)
    arguments = ", ".join(["0"] * arity)
    runner.replace_tests(prompt_id_for(question), [f"assert {name}({arguments}) == 999"])
    outcome = runner.run(prompt, source)
    assert outcome.error_class == "wrong_answer"
