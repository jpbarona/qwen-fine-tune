#!/usr/bin/env bash
# Run one reward arm or the MLflow dashboard inside the training image.
#
# Usage:
#   ./scripts/docker-run.sh <code-score|hybrid|gated-judge|gated-process|dashboard> [-- extra args]
#
# The repository is mounted at /app, so a code change needs no rebuild. The
# container gets the GPU, host shared memory, the host Docker socket for the
# sandboxed test runner, and a cache volume for the compiled kernels. The
# dashboard arm prints the remote store URL and starts nothing. The offline
# server is on bare metal through `rlvr dashboard --local`.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: docker-run.sh <code-score|hybrid|gated-judge|gated-process|dashboard> [-- extra args]" >&2
  exit 2
fi

# A container cannot start without docker access. Say which command fixes it
# instead of surfacing the daemon's permission error.
if ! docker info >/dev/null 2>&1; then
  echo "docker-run.sh: $(id -un) cannot reach the docker daemon. run:" >&2
  echo "  newgrp docker    # or open a new SSH session" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${RLVR_IMAGE:-ghcr.io/jpbarona/rlvr:latest}"
image_ref="$(docker image inspect "$image" --format '{{index .RepoDigests 0}}' 2>/dev/null || printf '%s' "$image")"

arm="$1"
shift
if [ "${1:-}" = "--" ]; then
  shift
fi

case "$arm" in
  code-score|hybrid|gated-judge|gated-process) ;;
  dashboard)
    echo "http://127.0.0.1:8081"
    exit 0
    ;;
  *)
    echo "docker-run.sh: unknown arm '$arm'" >&2
    exit 2
    ;;
esac

has_config=false
for arg in "$@"; do
  case "$arg" in
    --config|--config=*) has_config=true ;;
  esac
done
if [ "$has_config" = false ]; then
  set -- --config configs/real.toml "$@"
fi

# The test runner starts containers through the host daemon, so every path it
# hands to `docker run` is resolved on the host, not inside this container. The
# task directory must therefore live at a host path that is also mounted here
# under the same absolute path. `TMPDIR` points the runner at that directory.
runner_tmp="$repo_root/.runner-tmp"
mkdir -p "$runner_tmp"

# The container runs as root and imports the repository from the mount. Python
# would write root-owned __pycache__ folders into that mount and a later rsync
# by the host user would fail. No bytecode is written for this process tree.
exec docker run --rm --gpus all --ipc=host \
  -v "$repo_root":/app \
  -v "$repo_root":"$repo_root" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v rlvr-kernel-cache:/root/.cache \
  -e RLVR_VENV=/opt/venv \
  -e RLVR_IMAGE_REF="$image_ref" \
  -e TMPDIR="$runner_tmp" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e GIT_CONFIG_COUNT=1 \
  -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0='*' \
  -e CC=gcc \
  -e CXX=g++ \
  -e OPENROUTER_API_KEY \
  -e OPENCODE_GO_API_KEY \
  -e MLFLOW_TRACKING_URI \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e MLFLOW_S3_ENDPOINT_URL \
  -e AWS_DEFAULT_REGION \
  -w /app \
  "$image" /app/run.sh "$arm" "$@"