"""Tests for the packaging: the one runtime extra and the shipped configs."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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

    assert extras["tracking"] == ["mlflow==3.18.0"]


def test_no_extra_pins_torch_via_a_range() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    vllm = pyproject["project"]["optional-dependencies"]["vllm"]
    assert all(">=" not in requirement and "<" not in requirement for requirement in vllm)


def test_requires_python_is_3_12_or_newer() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.12"
