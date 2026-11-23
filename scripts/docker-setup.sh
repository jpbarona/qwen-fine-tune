#!/usr/bin/env bash
# Prepare a fresh GPU host for training in a container.
#
# Usage:
#   ./scripts/docker-setup.sh
#
# Run once on a new box. It installs Docker and the NVIDIA container runtime if
# they are missing, logs in to the image registry, and pulls the training image
# plus the small Python image the test runner uses.
#
# The registry login uses two environment variables:
#   GHCR_USER   the GitHub user name, default jpbarona
#   GHCR_TOKEN  a token with read:packages
#
# If the image is public, the login step is skipped.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${RLVR_IMAGE:-ghcr.io/jpbarona/rlvr:latest}"
registry="${image%%/*}"
test_image="python:3.12-slim"

need_sudo=""
if ! docker info >/dev/null 2>&1; then
  need_sudo="sudo -n"
fi

# The MLflow server runs on the VM and the box reaches it through an SSH reverse
# tunnel. The tunnel binds the Docker bridge gateway on the box so the training
# container can reach it. sshd refuses a non-loopback bind unless GatewayPorts
# allows it, so a missing line and a `GatewayPorts no` line both need fixing.
# The drop-in is included before the main file on Ubuntu, so it wins. Any other
# GatewayPorts line is rewritten too, because an earlier drop-in would win over
# ours.
ensure_gatewayports() {
  local drop_in=/etc/ssh/sshd_config.d/99-rlvr-gatewayports.conf
  local line="GatewayPorts clientspecified"
  local sudo_cmd=()
  local file

  if [ "$(id -u)" -eq 0 ]; then
    sudo_cmd=()
  elif [ -n "$need_sudo" ]; then
    sudo_cmd=($need_sudo)
  else
    sudo_cmd=(sudo -n)
  fi

  "${sudo_cmd[@]}" mkdir -p /etc/ssh/sshd_config.d 2>/dev/null || true
  if ! printf '%s\n' "$line" | "${sudo_cmd[@]}" tee "$drop_in" >/dev/null 2>&1; then
    echo "cannot write $drop_in, so the tunnel bind stays refused. run:"
    echo "  echo '$line' | sudo tee $drop_in"
    echo "  sudo sed -i -E 's|^[[:space:]]*GatewayPorts.*|$line|' \\"
    echo "    /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf"
    echo "  sudo sshd -t && sudo systemctl reload ssh"
    return 0
  fi

  for file in /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf; do
    [ -f "$file" ] || continue
    [ "$file" = "$drop_in" ] && continue
    if grep -qE '^[[:space:]]*GatewayPorts([[:space:]]|$)' "$file" 2>/dev/null; then
      "${sudo_cmd[@]}" sed -i -E \
        "s|^[[:space:]]*GatewayPorts([[:space:]].*)?\$|$line|" "$file"
      echo "replaced the GatewayPorts line in $file"
    fi
  done

  if ! "${sudo_cmd[@]}" sshd -t; then
    echo "docker-setup.sh: sshd -t rejected the configuration, sshd not reloaded" >&2
    exit 1
  fi
  "${sudo_cmd[@]}" systemctl reload ssh 2>/dev/null ||
    "${sudo_cmd[@]}" systemctl reload sshd
  echo "set $line and reloaded sshd"
}

ensure_gatewayports

if ! command -v docker >/dev/null 2>&1; then
  echo "installing docker"
  $need_sudo apt-get update -qq
  $need_sudo apt-get install -y -qq docker.io
fi

if ! dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
  echo "installing the NVIDIA container toolkit"
  curl -fsSL -o /tmp/nvidia-ctk.list \
    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list
  $need_sudo cp /tmp/nvidia-ctk.list /etc/apt/sources.list.d/nvidia-container-toolkit.list
  $need_sudo apt-get update -qq
  $need_sudo apt-get install -y -qq nvidia-container-toolkit
  $need_sudo nvidia-ctk runtime configure --runtime=docker
  $need_sudo systemctl restart docker
fi

if docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia"; then
  echo "nvidia runtime is configured"
else
  echo "warning: the nvidia runtime is not in docker info"
fi

if [ -n "${GHCR_TOKEN:-}" ]; then
  echo "$GHCR_TOKEN" | docker login "$registry" \
    -u "${GHCR_USER:-jpbarona}" --password-stdin
fi

echo "pulling $image"
docker pull "$image"

echo "pulling $test_image"
docker pull "$test_image"

echo
echo "READY. start training with:"
echo "  ./scripts/docker-run.sh code-score --config configs/lfm-spread.toml --runs-root runs"