#!/usr/bin/env bash
# Launch one reward arm with no manual edits.
#
# Usage:
#   ./run.sh <code-score|hybrid|gated-judge|gated-process> [-- extra args]
#
# The judge key is read from the OPENROUTER_API_KEY environment variable by the
# judge client, because the judge uses OpenRouter by default. The opencode
# provider reads OPENCODE_GO_API_KEY instead. The key is never written to this
# file, a config, or a run folder.
# Extra arguments are passed through to `rlvr run`, so a smoke run is:
#
#   ./run.sh code-score --config configs/smoke.toml
#
# Resume from the last checkpoint after a crash:
#
#   ./run.sh code-score --resume
#
# Select another interpreter with RLVR_VENV. The default is .venv.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: run.sh <code-score|hybrid|gated-judge|gated-process> [-- extra args]" >&2
  exit 2
fi

arm="$1"
shift
if [ "${1:-}" = "--" ]; then
  shift
fi

case "$arm" in
  code-score|hybrid|gated-judge|gated-process) ;;
  *)
    echo "run.sh: unknown arm '$arm'" >&2
    exit 2
    ;;
esac

# Only add the default config when the caller did not supply one.
has_config=false
for arg in "$@"; do
  case "$arg" in
    --config|--config=*) has_config=true ;;
  esac
done
if [ "$has_config" = false ]; then
  set -- --config configs/real.toml "$@"
fi

rlvr_venv="${RLVR_VENV:-.venv}"

# vLLM compiles kernels at the first step. The toolkit version must match the
# torch wheel, or nvcc rejects the torch headers. Prefer the exact match in
# /usr/local, then the copy inside the venv.
if [ -z "${CUDA_HOME:-}" ]; then
  torch_cuda="$("$rlvr_venv/bin/python" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
  if [ -n "$torch_cuda" ] && [ -x "/usr/local/cuda-$torch_cuda/bin/nvcc" ]; then
    CUDA_HOME="/usr/local/cuda-$torch_cuda"
  else
    for candidate in "$rlvr_venv"/lib/python*/site-packages/nvidia/cu*/bin/nvcc; do
      [ -x "$candidate" ] || continue
      CUDA_HOME="$(dirname "$(dirname "$candidate")")"
      break
    done
  fi
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_HOME
  case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
  esac
fi

# torch passes CC to nvcc as the host compiler. A plain gcc whose major version
# has no g++ partner cannot compile C++, so name g++ directly.
if [ -z "${CC:-}" ]; then
  export CC=g++
fi
if [ -z "${CXX:-}" ]; then
  export CXX=g++
fi

# Call the module, not the console script. The image's console script was
# generated at build time and still names the pre-refactor import path. The
# module path follows the mounted source, so a code move cannot break the run.
exec "$rlvr_venv/bin/python" -m rlvr.app.cli run --arm "$arm" "$@"
