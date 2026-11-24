"""Tests for the fresh-box setup script.

Every command the script touches is a stub on a temporary PATH, so no test
needs root, a docker daemon, apt or a network. The stubs record their argv and
answer from `FAKE_*` settings, which lets one test model a caller with sudo and
no docker group, another a caller with neither, and so on.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "docker-setup.sh"
USER = "ubuntu"

ID_STUB = """#!/usr/bin/env bash
case "${1:-}" in
  -u) printf '%s\\n' "${FAKE_UID:-1000}" ;;
  -un) printf '%s\\n' "${FAKE_USER:-ubuntu}" ;;
  -nG) printf '%s\\n' "${FAKE_GROUPS:-ubuntu}" ;;
  *) exec /usr/bin/id "$@" ;;
esac
"""

DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "${1:-}" = "info" ]; then
  ok="${FAKE_DOCKER_USER_OK:-0}"
  if [ "${FAKE_AS_ROOT:-0}" = 1 ]; then ok="${FAKE_DOCKER_ROOT_OK:-1}"; fi
  if [ "${FAKE_AS_SG:-0}" = 1 ]; then ok="${FAKE_DOCKER_SG_OK:-1}"; fi
  if [ "$ok" != 1 ]; then
    printf 'permission denied while trying to connect to the Docker daemon socket\\n' >&2
    exit 1
  fi
  if [ "${FAKE_DOCKER_RUNTIME_NVIDIA:-1}" = 1 ]; then
    printf 'Server:\\n Runtimes: io.containerd.runc.v2 nvidia runc\\n'
  else
    printf 'Server:\\n Runtimes: io.containerd.runc.v2 runc\\n'
  fi
  exit 0
fi
exit 0
"""

SUDO_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_SUDO_LOG"
if [ "${FAKE_SUDO_OK:-1}" != 1 ]; then
  printf 'sudo: a password is required\\n' >&2
  exit 1
fi
if [ "${1:-}" = "-n" ]; then shift; fi
if [ "${1:-}" = "true" ]; then exit 0; fi
FAKE_AS_ROOT=1 "$@"
"""

SG_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_SG_LOG"
if [ "${FAKE_SG_OK:-1}" != 1 ]; then
  printf 'sg: cannot open session\\n' >&2
  exit 1
fi
shift || true
if [ "${1:-}" = "-c" ]; then shift; fi
FAKE_AS_SG=1 /bin/sh -c "${1:-}"
"""

GETENT_STUB = """#!/usr/bin/env bash
if [ "${1:-}" = "group" ] && [ "${2:-}" = "docker" ]; then
  printf 'docker:x:999:%s\\n' "${FAKE_DOCKER_MEMBERS:-}"
fi
exit 0
"""

DPKG_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_DPKG_LOG"
printf '%s' "${FAKE_TOOLKIT_STATUS:-install ok installed}"
"""

TEE_STUB = """#!/usr/bin/env bash
cat >/dev/null
exit 0
"""

SIMPLE_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_LOG"
exit 0
"""


def write_stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def fake_environment(tmp_path: Path, settings: dict[str, str]) -> dict[str, str]:
    """A PATH where every host command is a stub, plus the chosen answers."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_stub(bin_dir, "id", ID_STUB)
    write_stub(bin_dir, "docker", DOCKER_STUB)
    write_stub(bin_dir, "sudo", SUDO_STUB)
    write_stub(bin_dir, "sg", SG_STUB)
    write_stub(bin_dir, "getent", GETENT_STUB)
    write_stub(bin_dir, "dpkg-query", DPKG_STUB)
    write_stub(bin_dir, "tee", TEE_STUB)
    for name in (
        "apt-get",
        "cp",
        "curl",
        "mkdir",
        "nvidia-ctk",
        "sed",
        "service",
        "sleep",
        "sshd",
        "systemctl",
        "usermod",
    ):
        write_stub(bin_dir, name, SIMPLE_STUB)

    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["HOME"] = str(tmp_path / "home")
    environment.pop("GHCR_TOKEN", None)
    environment.pop("GHCR_USER", None)
    environment.pop("RLVR_IMAGE", None)
    for name in (
        "FAKE_AS_ROOT",
        "FAKE_AS_SG",
        "FAKE_DOCKER_MEMBERS",
        "FAKE_DOCKER_RUNTIME_NVIDIA",
        "FAKE_DOCKER_SG_OK",
        "FAKE_DOCKER_USER_OK",
        "FAKE_SUDO_OK",
        "FAKE_TOOLKIT_STATUS",
    ):
        environment.pop(name, None)
    environment["FAKE_LOG"] = str(tmp_path / "commands.log")
    environment["FAKE_DOCKER_LOG"] = str(tmp_path / "docker.log")
    environment["FAKE_SUDO_LOG"] = str(tmp_path / "sudo.log")
    environment["FAKE_SG_LOG"] = str(tmp_path / "sg.log")
    environment["FAKE_DPKG_LOG"] = str(tmp_path / "dpkg.log")
    environment.update(settings)
    return environment


def run_setup(
    tmp_path: Path, settings: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=fake_environment(tmp_path, settings),
        cwd=REPO_ROOT,
        timeout=60,
    )


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def no_nvidia_ctk(bin_dir: Path) -> None:
    (bin_dir / "nvidia-ctk").unlink()


def test_a_sudo_caller_without_docker_access_is_added_and_pulls_through_sg(
    tmp_path,
) -> None:
    """The observed fresh-box case: sudo works, the docker group is missing."""
    result = run_setup(
        tmp_path,
        {
            "FAKE_SUDO_OK": "1",
            "FAKE_DOCKER_USER_OK": "0",
            "FAKE_DOCKER_ROOT_OK": "1",
            "FAKE_DOCKER_SG_OK": "1",
            "GHCR_TOKEN": "token",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "adding ubuntu to the docker group" in result.stdout
    assert "already installed" in result.stdout
    assert "READY" in result.stdout
    assert "newgrp docker" in result.stdout
    # No bare daemon permission error reaches the caller.
    assert "permission denied" not in result.stderr

    assert "-aG docker ubuntu" in read(tmp_path / "sudo.log")
    sg_log = read(tmp_path / "sg.log")
    assert "docker login ghcr.io" in sg_log
    assert "pull ghcr.io/jpbarona/rlvr:latest" in sg_log
    assert "pull python:3.12-slim" in sg_log


def test_a_caller_with_no_root_and_no_group_stops_before_the_pull(tmp_path) -> None:
    result = run_setup(
        tmp_path,
        {
            "FAKE_SUDO_OK": "0",
            "FAKE_DOCKER_USER_OK": "0",
            "GHCR_TOKEN": "token",
        },
    )

    assert result.returncode != 0
    assert "sudo usermod -aG docker ubuntu" in result.stderr
    assert "newgrp docker" in result.stderr
    assert "./scripts/docker-setup.sh" in result.stderr
    docker_log = read(tmp_path / "docker.log")
    assert "login" not in docker_log
    assert "pull" not in docker_log


def test_an_installed_toolkit_is_not_reinstalled(tmp_path) -> None:
    """`dpkg -l | grep` reported a present toolkit as missing under pipefail."""
    result = run_setup(
        tmp_path,
        {
            "FAKE_SUDO_OK": "0",
            "FAKE_DOCKER_USER_OK": "1",
            "FAKE_TOOLKIT_STATUS": "install ok installed",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "the NVIDIA container toolkit is already installed" in result.stdout
    assert "installing the NVIDIA container toolkit" not in result.stdout
    dpkg_log = read(tmp_path / "dpkg.log")
    assert "-W" in dpkg_log
    assert "-l" not in dpkg_log
    # apt must not run without root, and must not run here at all.
    assert read(tmp_path / "commands.log") == ""


def test_a_missing_toolkit_without_root_prints_the_apt_commands(tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    environment = fake_environment(
        tmp_path,
        {
            "FAKE_SUDO_OK": "0",
            "FAKE_DOCKER_USER_OK": "1",
            "FAKE_TOOLKIT_STATUS": "unknown ok not-installed",
        },
    )
    no_nvidia_ctk(bin_dir)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
        timeout=60,
    )

    assert result.returncode != 0
    assert "sudo apt-get install -y nvidia-container-toolkit" in result.stderr
    assert "curl -fsSL" in result.stderr
    assert "pull" not in read(tmp_path / "docker.log")


def test_a_group_in_the_database_but_not_the_session_uses_sg(tmp_path) -> None:
    result = run_setup(
        tmp_path,
        {
            "FAKE_SUDO_OK": "0",
            "FAKE_DOCKER_USER_OK": "0",
            "FAKE_DOCKER_MEMBERS": "ubuntu",
            "FAKE_SG_OK": "0",
        },
    )

    assert result.returncode != 0
    assert "this session cannot use the docker group yet" in result.stderr
    assert "newgrp docker" in result.stderr
    assert "pull" not in read(tmp_path / "docker.log")


def test_an_unconfigured_runtime_without_root_prints_both_commands(tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    environment = fake_environment(
        tmp_path,
        {
            "FAKE_SUDO_OK": "0",
            "FAKE_DOCKER_USER_OK": "1",
            "FAKE_DOCKER_RUNTIME_NVIDIA": "0",
        },
    )
    no_nvidia_ctk(bin_dir)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
        timeout=60,
    )

    assert result.returncode != 0
    assert "sudo nvidia-ctk runtime configure --runtime=docker" in result.stderr
    assert "sudo systemctl restart docker" in result.stderr
    assert "pull" not in read(tmp_path / "docker.log")
