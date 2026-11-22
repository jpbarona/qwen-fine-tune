#!/usr/bin/env bash
# Set up this host for training in one command.
#
# Usage:
#   ./setup.sh            install everything that is missing
#   ./setup.sh --check    report only, change nothing
#
# Run it once on a fresh GPU host. Running it again is safe and fast, because
# every step skips work that is already done.
#
# It installs the Python 3.12 runtime, the compiler and headers, the venv, the
# pinned CUDA 13 torch and vLLM stack, the fast convolution kernel, and the
# screen file the arm configs need. It also points the build at the GPU in this
# host, so the kernels compile for one architecture and the run is fast.
#
# After it prints READY, start training with:
#   ./run.sh code-score --config configs/lfm-spread.toml --runs-root runs

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

echo "=== rlvr setup ==="
exec "$repo_root/scripts/setup-env.sh" "$@"
