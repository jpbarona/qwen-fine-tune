"""Tests for the reverse tunnel and the remote launch script.

The scripts are driven with stub `ssh`, `curl` and `docker` binaries on a
temporary PATH. No test reaches a real host. The tunnel loop is killed from the
pid file it writes, so no background process outlives a test.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST = "gpu-box.example"

SSH_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$FAKE_SSH_LOG"
exit 0
"""

DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$FAKE_DOCKER_LOG"
exit 0
"""


def write_stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def fake_environment(
    tmp_path: Path, *, curl_exit: int, ssh_log: Path | None = None
) -> dict[str, str]:
    """A PATH whose ssh, curl and docker are stubs, plus fast health settings."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    write_stub(bin_dir, "ssh", SSH_STUB)
    write_stub(bin_dir, "docker", DOCKER_STUB)
    write_stub(bin_dir, "curl", f"#!/usr/bin/env bash\nexit {curl_exit}\n")

    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["FAKE_SSH_LOG"] = str(ssh_log or tmp_path / "ssh.log")
    environment["FAKE_DOCKER_LOG"] = str(tmp_path / "docker.log")
    environment["SSH_KEY"] = str(tmp_path / "id_ed25519")
    environment["RLVR_TUNNEL_DIR"] = str(tmp_path / "cache")
    # The real defaults are 30 tries at 2 seconds. The test must not wait a
    # minute for the dead end it is testing.
    environment["RLVR_HEALTH_ATTEMPTS"] = "2"
    environment["RLVR_HEALTH_INTERVAL"] = "0"
    # A stub ssh exits at once. Park the reconnect loop instead of spinning.
    environment["RLVR_TUNNEL_RETRY_DELAY"] = "3600"
    return environment


def start(
    tmp_path: Path, *args: str, curl_exit: int, ssh_log: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "start-remote-run.sh"),
            *args,
        ],
        capture_output=True,
        text=True,
        env=fake_environment(tmp_path, curl_exit=curl_exit, ssh_log=ssh_log),
        cwd=REPO_ROOT,
        timeout=60,
    )


def run_script_with_fake_binaries(
    tmp_path: Path, log: Path
) -> subprocess.CompletedProcess[str]:
    return start(tmp_path, "code-score", HOST, curl_exit=1, ssh_log=log)


def run_script_with_dead_health(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return start(tmp_path, "code-score", HOST, curl_exit=1)


def stop_tunnel(tmp_path: Path) -> None:
    """Kill the tunnel loop started by open-tunnel.sh, if one is running."""
    for pid_file in (tmp_path / "cache").glob("tunnel-*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def test_start_remote_run_uses_a_reverse_forward(tmp_path) -> None:
    log = tmp_path / "ssh.log"
    try:
        run_script_with_fake_binaries(tmp_path, log)
        assert log.is_file()
        text = log.read_text(encoding="utf-8")
        assert "-R" in text
        assert "GatewayPorts" not in text
    finally:
        stop_tunnel(tmp_path)


def test_start_remote_run_fails_when_health_never_answers(tmp_path) -> None:
    try:
        result = run_script_with_dead_health(tmp_path)
        assert result.returncode != 0
        assert "health" in result.stderr.lower()
    finally:
        stop_tunnel(tmp_path)


def test_start_remote_run_launches_the_docker_runner_when_health_answers(
    tmp_path,
) -> None:
    try:
        result = start(tmp_path, "code-score", HOST, "--fake", curl_exit=0)
        assert result.returncode == 0, result.stderr
        docker_log = tmp_path / "docker.log"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (
            docker_log.is_file() and "run" in docker_log.read_text(encoding="utf-8")
        ):
            time.sleep(0.05)
        text = docker_log.read_text(encoding="utf-8")
        assert "run" in text
        assert "--fake" in text
    finally:
        stop_tunnel(tmp_path)
