#!/usr/bin/env bash
# Run one reward arm inside the training image.
#
# Usage:
#   ./scripts/docker-run.sh <arm> [-- extra args]
#
# The repository is mounted at /app, so a code change needs no rebuild. The
# container gets the GPU, host shared memory, the host Docker socket for the
# sandboxed test runner, and a cache volume for the compiled kernels.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: docker-run.sh <code-score|hybrid|gated-judge|gated-process> [-- extra args]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${RLVR_IMAGE:-ghcr.io/jpbarona/rlvr:latest}"

arm="$1"
shift
if [ "${1:-}" = "--" ]; then
  shift
fi

case "$arm" in
  code-score|hybrid|gated-judge|gated-process) ;;
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

exec docker run --rm --gpus all --ipc=host \
  -v "$repo_root":/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v rlvr-kernel-cache:/root/.cache \
  -e RLVR_VENV=/opt/venv \
  -e OPENROUTER_API_KEY \
  -e OPENCODE_GO_API_KEY \
  -w /app \
  "$image" /app/run.sh "$arm" "$@"