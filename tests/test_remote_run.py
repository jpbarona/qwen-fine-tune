"""Tests for the reverse tunnel and the remote launch script.

The scripts are driven with stub `ssh`, `curl` and `docker` binaries on a
temporary PATH. No test reaches a real host. The stub `ssh` answers the gateway
question and the health probe, and the `curl` stub logs any local call so a
regression to a local health check fails. The tunnel loop is killed from the
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
GATEWAY = "10.42.0.1"

SSH_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$FAKE_SSH_LOG"
gateway=""
probe=0
for arg in "$@"; do
  case "$arg" in
    *"network inspect"*) gateway="$FAKE_GATEWAY" ;;
    *"curl "*) probe=1 ;;
  esac
done
if [ -n "$gateway" ]; then
  printf '%s\\n' "$gateway"
fi
if [ "$probe" = 1 ]; then
  count=0
  if [ -s "$FAKE_SSH_COUNT" ]; then
    count="$(cat "$FAKE_SSH_COUNT")"
  fi
  count=$((count + 1))
  printf '%s\\n' "$count" >"$FAKE_SSH_COUNT"
  case "${FAKE_HEALTH_MODE:-ok}" in
    ok) exit 0 ;;
    dead) exit 1 ;;
    probe-only) if [ "$count" -eq 1 ]; then exit 0; fi; exit 1 ;;
    *) exit 1 ;;
  esac
fi
exit 0
"""

DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$FAKE_DOCKER_LOG"
exit 0
"""

CURL_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$FAKE_CURL_LOG"
exit 1
"""


def write_stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def fake_environment(
    tmp_path: Path, *, health_mode: str = "ok", ssh_log: Path | None = None
) -> dict[str, str]:
    """A PATH whose ssh, curl and docker are stubs, plus fast probe settings."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    write_stub(bin_dir, "ssh", SSH_STUB)
    write_stub(bin_dir, "docker", DOCKER_STUB)
    write_stub(bin_dir, "curl", CURL_STUB)

    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["FAKE_SSH_LOG"] = str(ssh_log or tmp_path / "ssh.log")
    environment["FAKE_SSH_COUNT"] = str(tmp_path / "ssh.count")
    environment["FAKE_GATEWAY"] = GATEWAY
    environment["FAKE_HEALTH_MODE"] = health_mode
    environment["FAKE_DOCKER_LOG"] = str(tmp_path / "docker.log")
    environment["FAKE_CURL_LOG"] = str(tmp_path / "curl.log")
    environment["SSH_KEY"] = str(tmp_path / "id_ed25519")
    environment["RLVR_TUNNEL_DIR"] = str(tmp_path / "cache")
    # The real defaults are 30 tries at 2 seconds. The test must not wait a
    # minute for the dead ends it is testing.
    environment["RLVR_PROBE_ATTEMPTS"] = "2"
    environment["RLVR_PROBE_INTERVAL"] = "0"
    environment["RLVR_HEALTH_ATTEMPTS"] = "2"
    environment["RLVR_HEALTH_INTERVAL"] = "0"
    # A stub ssh exits at once. Park the reconnect loop instead of spinning.
    environment["RLVR_TUNNEL_RETRY_DELAY"] = "3600"
    return environment


def run_script(
    tmp_path: Path,
    *args: str,
    health_mode: str = "ok",
    ssh_log: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "start-remote-run.sh"), *args],
        capture_output=True,
        text=True,
        env=fake_environment(tmp_path, health_mode=health_mode, ssh_log=ssh_log),
        cwd=REPO_ROOT,
        timeout=60,
    )


def run_open_tunnel(
    tmp_path: Path, *args: str, health_mode: str = "ok"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "open-tunnel.sh"), *args],
        capture_output=True,
        text=True,
        env=fake_environment(tmp_path, health_mode=health_mode),
        cwd=REPO_ROOT,
        timeout=60,
    )


def run_script_with_fake_binaries(
    tmp_path: Path, log: Path
) -> subprocess.CompletedProcess[str]:
    return run_script(tmp_path, "code-score", HOST, ssh_log=log)


def run_script_with_dead_health(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return run_script(tmp_path, "code-score", HOST, health_mode="probe-only")


def ssh_args(tmp_path: Path) -> list[str]:
    path = tmp_path / "ssh.log"
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def reverse_forwards(tmp_path: Path) -> list[str]:
    """Every token that follows a `-R` in the logged ssh argv."""
    lines = ssh_args(tmp_path)
    return [lines[index + 1] for index, line in enumerate(lines) if line == "-R"]


def wait_for_reverse_forward(tmp_path: Path, timeout: float = 5.0) -> list[str]:
    """The loop logs its ssh call after open-tunnel.sh returns, so poll for it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        forwards = reverse_forwards(tmp_path)
        if forwards:
            return forwards
        time.sleep(0.02)
    return reverse_forwards(tmp_path)


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
        result = run_script_with_fake_binaries(tmp_path, log)
        assert result.returncode == 0, result.stderr
        assert wait_for_reverse_forward(tmp_path) == [
            f"{GATEWAY}:8081:127.0.0.1:8081"
        ]
        assert "GatewayPorts" not in "\n".join(ssh_args(tmp_path))
        # The health check goes through the box against the forwarded address.
        probe = [line for line in ssh_args(tmp_path) if "curl " in line]
        assert probe
        assert set(probe) == {f"curl -sSf http://{GATEWAY}:8081/health"}
        # No local curl, which would pass with a dead forward.
        assert not (tmp_path / "curl.log").exists()
    finally:
        stop_tunnel(tmp_path)


def test_open_tunnel_publishes_the_gateway_state(tmp_path) -> None:
    try:
        result = run_open_tunnel(tmp_path, HOST)
        assert result.returncode == 0, result.stderr
        state = tmp_path / "cache" / f"tunnel-{HOST}-8081.state"
        assert state.read_text(encoding="utf-8") == f"gateway={GATEWAY}\nport=8081\n"
        pid = int((tmp_path / "cache" / f"tunnel-{HOST}-8081.pid").read_text())
        assert pid > 0
    finally:
        stop_tunnel(tmp_path)


def test_a_second_call_reuses_the_tunnel(tmp_path) -> None:
    try:
        first = run_script(tmp_path, "code-score", HOST)
        assert first.returncode == 0, first.stderr
        # The user prefix is not part of the cache key.
        second = run_script(tmp_path, "code-score", f"Ubuntu@{HOST}")
        assert second.returncode == 0, second.stderr
        assert "already open" in second.stdout
        assert wait_for_reverse_forward(tmp_path) == [
            f"{GATEWAY}:8081:127.0.0.1:8081"
        ]
    finally:
        stop_tunnel(tmp_path)


def test_a_dead_open_tunnel_is_not_reused(tmp_path) -> None:
    try:
        first = run_script(tmp_path, "code-score", HOST)
        assert first.returncode == 0, first.stderr
        second = run_script(tmp_path, "code-score", HOST, health_mode="dead")
        assert second.returncode != 0
        assert "does not answer" in second.stderr
        assert wait_for_reverse_forward(tmp_path) == [
            f"{GATEWAY}:8081:127.0.0.1:8081"
        ]
    finally:
        stop_tunnel(tmp_path)


def test_open_tunnel_ignores_a_stale_pid_file(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    pid_file = cache / f"tunnel-{HOST}-8081.pid"
    state_file = cache / f"tunnel-{HOST}-8081.state"
    pid_file.write_text("4194304\n", encoding="utf-8")
    state_file.write_text("gateway=10.0.0.9\nport=8081\n", encoding="utf-8")
    try:
        result = run_open_tunnel(tmp_path, HOST)
        assert result.returncode == 0, result.stderr
        assert "already open" not in result.stdout
        assert pid_file.read_text(encoding="utf-8").strip() != "4194304"
        assert state_file.read_text(encoding="utf-8") == (
            f"gateway={GATEWAY}\nport=8081\n"
        )
    finally:
        stop_tunnel(tmp_path)


def test_open_tunnel_fails_when_the_forward_never_answers(tmp_path) -> None:
    try:
        result = run_open_tunnel(tmp_path, HOST, health_mode="dead")
        assert result.returncode != 0
        assert f"{GATEWAY}:8081" in result.stderr
        cache = tmp_path / "cache"
        assert not list(cache.glob("tunnel-*.pid"))
        assert not list(cache.glob("tunnel-*.state"))
    finally:
        stop_tunnel(tmp_path)


def test_start_remote_run_fails_when_the_box_health_never_answers(tmp_path) -> None:
    try:
        result = run_script_with_dead_health(tmp_path)
        assert result.returncode != 0
        assert "health" in result.stderr.lower()
        assert not (tmp_path / "docker.log").exists()
    finally:
        stop_tunnel(tmp_path)
