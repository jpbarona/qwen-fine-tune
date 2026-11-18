#!/usr/bin/env bash
# Create the isolated LFM2.5 environment.
#
# Usage:
#   ./scripts/setup-lfm-env.sh [venv path]
#
# Default venv path is .venv-lfm at the repository root. The environment is
# separate from .venv because the LFM extra pins Transformers 5, and the Qwen
# vLLM extra pins a Transformers 4 stack.
#
# torch comes from the cu128 index first, then the `lfm` extra is installed
# against it. Running the script twice is safe: an existing venv is reused and
# pip reports the pinned versions as already satisfied.
#
# Some hosts ship Python 3.12 without ensurepip. In that case the venv is
# created with --without-pip and pip is bootstrapped from the system pip.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${1:-$repo_root/.venv-lfm}"

case "$venv" in
  /*) ;;
  *) venv="$PWD/$venv" ;;
esac

python_bin="${PYTHON:-python3}"
torch_index="https://download.pytorch.org/whl/cu128"
torch_version="2.7.1"

echo "repository: $repo_root"
echo "venv:       $venv"

if [ ! -x "$venv/bin/python" ]; then
  echo "creating venv with $("$python_bin" -V 2>&1)"
  if ! "$python_bin" -m venv "$venv" 2>/dev/null; then
    rm -rf "$venv"
    "$python_bin" -m venv --without-pip "$venv"
    if ! "$python_bin" -m pip --python "$venv/bin/python" install -q pip setuptools wheel; then
      echo "setup-lfm-env.sh: could not bootstrap pip into $venv" >&2
      echo "install python3-venv or a pip for $("$python_bin" -V 2>&1)" >&2
      exit 1
    fi
  fi
else
  echo "venv already exists, reusing it"
fi

if ! "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "bootstrapping pip"
  "$python_bin" -m pip --python "$venv/bin/python" install -q pip setuptools wheel
fi

echo "installing torch==$torch_version from $torch_index"
"$venv/bin/python" -m pip install "torch==$torch_version" --index-url "$torch_index"

echo "installing -e .[lfm]"
"$venv/bin/python" -m pip install -e "$repo_root[lfm]"

echo "installing pytest"
"$venv/bin/python" -m pip install "pytest>=9,<10"

echo "versions"
"$venv/bin/python" - <<'PY'
import torch
import transformers
import peft
import accelerate
import safetensors

print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"peft         {peft.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"safetensors  {safetensors.__version__}")
print(f"cuda         {torch.cuda.is_available()}")
PY

echo "done. use the environment with:"
echo "  RLVR_VENV=$venv ./run.sh code-score --config configs/lfm-smoke.toml --runs-root runs-lfm-smoke"
