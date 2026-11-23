"""Tests for the packaging: the one runtime extra and the shipped configs."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_backup_module() -> ModuleType:
    path = REPO_ROOT / "deploy" / "backup.py"
    spec = importlib.util.spec_from_file_location("deploy_backup_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_vllm_extra_pins_the_whole_gpu_stack() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    expected = {
        "vllm": "0.34.0",
        "torch": "2.14.0",
        "transformers": "5.22.2",
        "peft": "0.18.1",
        "accelerate": "1.8.1",
    }
    pins = {}
    for requirement in extras["vllm"]:
        name, _, version = requirement.partition("==")
        assert version, f"{requirement!r} must pin an exact version"
        pins[name] = version
    assert pins == expected


def test_the_old_environment_extras_are_gone() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras) == {"dev", "tracking", "vllm"}


def test_the_tracking_extra_pins_mlflow() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    expected = {"mlflow": "3.18.0", "boto3": "1.44.9"}
    pins = {}
    for requirement in extras["tracking"]:
        name, _, version = requirement.partition("==")
        assert version, f"{requirement!r} must pin an exact version"
        pins[name] = version
    assert pins == expected


def test_no_extra_pins_torch_via_a_range() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    vllm = pyproject["project"]["optional-dependencies"]["vllm"]
    assert all(">=" not in requirement and "<" not in requirement for requirement in vllm)


def test_requires_python_is_3_12_or_newer() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.12"


def test_service_runs_the_repo_venv() -> None:
    text = (REPO_ROOT / "deploy" / "mlflow.service").read_text(encoding="utf-8")
    assert "/home/opencode/Work/rl-grpo/.venv/bin/mlflow" in text
    assert "--workers 1" in text
    assert "127.0.0.1" in text


def test_env_example_lists_the_bucket_variables() -> None:
    text = (REPO_ROOT / "deploy" / "mlflow.env.example").read_text(encoding="utf-8")
    for name in (
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ):
        assert name in text


def test_backup_script_is_importable_and_has_a_main() -> None:
    module = _load_backup_module()
    assert callable(module.main)
