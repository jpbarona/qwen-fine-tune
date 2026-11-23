#!/usr/bin/env bash
# Create the runtime environment.
#
# Usage:
#   ./scripts/setup-env.sh [--check] [--strict] [venv path]
#
# Default venv path is .venv at the repository root. One extra, `vllm`,
# installs vLLM, torch, Transformers, PEFT and Accelerate at pins that work
# together for Qwen and LFM2.5. Running the script twice is safe: an existing
# venv is reused and pip reports the pinned versions as already satisfied.
#
# A preflight runs first and fails fast with the exact fix command. It checks
# the interpreter version, the Python development headers, ninja, the CUDA
# toolkit and the screen file that each arm config names. These are the checks
# that stop a run late, after the model has downloaded.
#
# `--check` runs the preflight only and changes nothing. `--strict` turns the
# warnings into errors, which is useful on a fresh host.
#
# Some hosts ship Python 3.12 without ensurepip. In that case the venv is
# created with --without-pip and pip is bootstrapped from the system pip.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
check_only=0
strict=0
install_cuda_flag=1
fast=0
venv=""

for arg in "$@"; do
  case "$arg" in
    --check) check_only=1 ;;
    --strict) strict=1 ;;
    --no-cuda) install_cuda_flag=0 ;;
    --fast) fast=1 ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      if [ -n "$venv" ]; then
        echo "setup-env.sh: only one venv path is allowed" >&2
        exit 2
      fi
      venv="$arg"
      ;;
  esac
done

venv="${venv:-$repo_root/.venv}"
case "$venv" in
  /*) ;;
  *) venv="$PWD/$venv" ;;
esac

WARNINGS=0
ERRORS=0

warn() {
  WARNINGS=$((WARNINGS + 1))
  echo "warn:  $*"
}

fail() {
  ERRORS=$((ERRORS + 1))
  echo "error: $*"
}

# The interpreter. PYTHON wins, then the newest python3.x that is 3.12 or newer.
find_python() {
  if [ -n "${PYTHON:-}" ]; then
    echo "$PYTHON"
    return 0
  fi
  local candidate version major minor
  for candidate in python3.13 python3.12 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version="$("$candidate" -c 'import sys; print("%d %d" % sys.version_info[:2])' 2>/dev/null)"
    [ -n "$version" ] || continue
    major="${version% *}"
    minor="${version#* }"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 12 ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

python_bin="$(find_python || true)"
if [ -z "$python_bin" ]; then
  fail "no Python 3.12 or newer found. The project needs 3.12."
  echo "  Ubuntu 24.04: sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
  echo "  Ubuntu 22.04: sudo add-apt-repository -y ppa:deadsnakes/ppa"
  echo "                sudo apt-get update"
  echo "                sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
  echo "  Then set PYTHON=python3.12 or put it on PATH."
  exit 1
fi

if ! "$python_bin" -c 'import sys' >/dev/null 2>&1; then
  echo "setup-env.sh: PYTHON=$python_bin cannot run" >&2
  exit 1
fi

py_version="$("$python_bin" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
py_major="${py_version%%.*}"
py_minor="${py_version#*.}"

if [ "$py_major" -ne 3 ] || [ "$py_minor" -lt 12 ]; then
  fail "Python $py_version is too old. The project needs 3.12."
  echo "  Ubuntu 24.04: sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
  echo "  Ubuntu 22.04: sudo add-apt-repository -y ppa:deadsnakes/ppa"
  echo "                sudo apt-get update"
  echo "                sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
  echo "  Then set PYTHON=python3.12."
  exit 1
fi

echo "repository: $repo_root"
echo "venv:       $venv"
echo "python:     $python_bin ($py_version)"

# Compile for one architecture only. The default builds for every supported GPU
# generation, which multiplies setup time and the first step.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
  case "$compute_cap" in
    [0-9].[0-9]) export TORCH_CUDA_ARCH_LIST="$compute_cap" ;;
  esac
fi
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  echo "arch:       $TORCH_CUDA_ARCH_LIST"
fi

# Development headers. Triton and torch inductor compile a small C extension at
# engine start, so a missing Python.h stops the run after the model download.
include_dir="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
headers_ok=0
if [ -f "$include_dir/Python.h" ]; then
  headers_ok=1
fi

# ninja lets vLLM build kernels. Without it FlashInfer stays disabled.
ninja_ok=0
if command -v ninja >/dev/null 2>&1; then
  ninja_ok=1
fi

host_packages=()
[ "$headers_ok" -eq 0 ] && host_packages+=("python${py_version}-dev")
[ "$ninja_ok" -eq 0 ] && host_packages+=("ninja-build")
# nvcc preprocesses through the host C++ compiler, so g++ is required, not just
# the C compiler. A host with gcc and no g++ fails inside nvcc with a bare
# "cc1plus: No such file or directory".
if ! command -v g++ >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then
  host_packages+=("build-essential")
fi

if [ "${#host_packages[@]}" -gt 0 ] && [ "$check_only" -eq 0 ]; then
  if sudo -n true 2>/dev/null; then
    echo "installing host packages: ${host_packages[*]}"
    if sudo -n apt-get update -qq && sudo -n apt-get install -y -qq "${host_packages[@]}"; then
      include_dir="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
      [ -f "$include_dir/Python.h" ] && headers_ok=1
      command -v ninja >/dev/null 2>&1 && ninja_ok=1
    else
      fail "could not install the host packages: ${host_packages[*]}"
    fi
  else
    echo "no passwordless sudo, install these by hand:"
    echo "  sudo apt-get install -y ${host_packages[*]}"
  fi
fi

if [ "$headers_ok" -eq 0 ]; then
  fail "Python headers are missing. Triton and torch inductor compile at engine start."
  echo "  sudo apt-get install -y python${py_version}-dev"
  echo "  headers were expected in $include_dir"
fi

if [ "$ninja_ok" -eq 0 ]; then
  warn "ninja is missing, so vLLM and FlashInfer kernels stay disabled."
  echo "  sudo apt-get install -y ninja-build"
fi

# The CUDA toolkit. The prebuilt wheels run without it, but the fast kernels
# need nvcc, and causal_conv1d has to compile. The check runs again after pip,
# when torch reports the exact CUDA version to match.
cuda_ok=0
if command -v nvcc >/dev/null 2>&1; then
  cuda_ok=1
elif [ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ] || [ -x /usr/local/cuda/bin/nvcc ]; then
  cuda_ok=1
else
  warn "no nvcc found. It will be installed after torch, matching the wheel."
fi

venv_cuda_prefix() {
  local candidate
  for candidate in "$venv"/lib/python*/site-packages/nvidia/cu*/bin/nvcc; do
    [ -x "$candidate" ] || continue
    dirname "$(dirname "$candidate")"
    return 0
  done
  return 1
}

# nvcc must match the CUDA version torch was built with. The venv toolkit can be
# newer, and mixing it with the torch headers fails inside nvcc with a version
# complaint. Prefer /usr/local/cuda-<torch version> when it exists.
matching_cuda_prefix() {
  local want="$1"
  local candidate
  case "$want" in
    [0-9]*.[0-9]*) ;;
    *) return 1 ;;
  esac
  for candidate in "/usr/local/cuda-${want}" "/usr/local/cuda-${want%%.*}"; do
    [ -x "$candidate/bin/nvcc" ] && { echo "$candidate"; return 0; }
  done
  return 1
}

install_cuda() {
  local want="$1"
  local major="${want%%.*}"
  local minor="${want#*.}"
  [ "$minor" = "$want" ] && minor=0
  local distro repo
  # shellcheck disable=SC1091
  . /etc/os-release
  distro="${ID}${VERSION_ID//./}"
  case "$distro" in
    ubuntu2204) repo=ubuntu2204 ;;
    ubuntu2404) repo=ubuntu2404 ;;
    debian12) repo=debian12 ;;
    *)
      warn "no NVIDIA apt repo for $distro. Install CUDA ${want} by hand."
      return 1
      ;;
  esac
  if ! sudo -n true 2>/dev/null; then
    warn "no passwordless sudo, so CUDA ${want} was not installed."
    echo "  install the CUDA ${major}.${minor} toolkit for ${repo} by hand."
    return 1
  fi
  local keyring=/tmp/cuda-keyring-${repo}.deb
  local url="https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "installing CUDA ${major}.${minor} toolkit from ${repo}"
  if ! curl -fsSL -o "$keyring" "$url"; then
    warn "could not download $url"
    return 1
  fi
  if ! sudo -n dpkg -i "$keyring" >/dev/null 2>&1 && ! sudo -n apt-get install -y -f -qq; then
    warn "could not install the CUDA keyring"
    return 1
  fi
  sudo -n apt-get update -qq || true
  if ! sudo -n apt-get install -y -qq "cuda-toolkit-${major}-${minor}"; then
    warn "could not install cuda-toolkit-${major}-${minor}"
    return 1
  fi
  rm -f "$keyring"
  export CUDA_HOME="/usr/local/cuda-${major}.${minor}"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "CUDA_HOME=$CUDA_HOME"
    echo "add this to your shell profile:"
    echo "  export CUDA_HOME=$CUDA_HOME"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    return 0
  fi
  warn "cuda-toolkit-${major}-${minor} installed but $CUDA_HOME/bin/nvcc is missing"
  return 1
}

# The screen. It is gitignored, so a fresh clone has none and each arm fails
# after the model loads.
screen_missing=0
shopt -s nullglob
for config in "$repo_root"/configs/*.toml; do
  policy="$(sed -n 's/^[[:space:]]*pool_policy[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$config" | head -n 1)"
  [ -n "$policy" ] || continue
  screen="$(sed -n 's/^[[:space:]]*screen_rows[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$config" | head -n 1)"
  [ -n "$screen" ] || continue
  case "$screen" in
    /*) screen_path="$screen" ;;
    *) screen_path="$repo_root/$screen" ;;
  esac
  if [ ! -f "$screen_path" ]; then
    screen_missing=$((screen_missing + 1))
    warn "$(basename "$config") wants $screen, which is not in git."
  fi
done
shopt -u nullglob

if [ "$screen_missing" -gt 0 ]; then
  echo "  restore the screen with:"
  echo "  rclone copy hetzner-dumps:opencodeagent1/dumps/screens/2026-11-21-lfm-base \\"
  echo "    \"$repo_root/screens/mbpp-base\""
  echo "  the two arm configs fail at load without it."
  if [ "$check_only" -eq 0 ] && command -v rclone >/dev/null 2>&1; then
    mkdir -p "$repo_root/screens/mbpp-base"
    if rclone copy hetzner-dumps:opencodeagent1/dumps/screens/2026-11-21-lfm-base \
        "$repo_root/screens/mbpp-base" >/dev/null 2>&1; then
      echo "  fetched the screen with rclone"
      screen_missing=0
    else
      echo "  rclone could not fetch it, check the remote name and the config"
    fi
  fi
fi

echo "checks: headers=$headers_ok ninja=$ninja_ok cuda=$cuda_ok screens_missing=$screen_missing errors=$ERRORS warnings=$WARNINGS"

if [ "$check_only" -eq 1 ]; then
  if [ "$ERRORS" -gt 0 ]; then
    exit 1
  fi
  if [ "$strict" -eq 1 ] && [ "$WARNINGS" -gt 0 ]; then
    exit 1
  fi
  exit 0
fi

if [ "$ERRORS" -gt 0 ]; then
  echo "setup-env.sh: fix the errors above, then run this script again." >&2
  exit 1
fi

if [ -x "$venv/bin/python" ]; then
  venv_version="$("$venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "broken")"
  if [ "$venv_version" != "$py_version" ] && [ "$venv_version" != "broken" ]; then
    echo "venv was created with Python $venv_version, not $py_version" >&2
    echo "remove it and run again: rm -rf \"$venv\"" >&2
    exit 1
  fi
fi

if [ ! -x "$venv/bin/python" ]; then
  echo "creating venv with $("$python_bin" -V 2>&1)"
  if ! "$python_bin" -m venv "$venv" 2>/dev/null; then
    rm -rf "$venv"
    "$python_bin" -m venv --without-pip "$venv"
    if ! "$python_bin" -m pip --python "$venv/bin/python" install -q pip setuptools wheel; then
      echo "setup-env.sh: could not bootstrap pip into $venv" >&2
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

echo "installing -e .[vllm]"
"$venv/bin/python" -m pip install -e "$repo_root[vllm]"

echo "installing pytest"
"$venv/bin/python" -m pip install "pytest>=9,<10"

# torch reports the CUDA version its wheel was built for. Install the matching
# toolkit when nvcc is missing, so the fast kernels build on the first try.
if [ "$install_cuda_flag" -eq 1 ] && [ "$cuda_ok" -eq 0 ]; then
  want="$("$venv/bin/python" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"

  # Order of preference: the toolkit that matches torch exactly, then the copy
  # inside the venv only when its version matches, then an apt install.
  matching="$(matching_cuda_prefix "$want" || true)"
  venv_cuda="$(venv_cuda_prefix || true)"

  if [ -n "$matching" ]; then
    CUDA_HOME="$matching"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    cuda_ok=1
  elif [ -n "$venv_cuda" ]; then
    venv_version="$("$venv_cuda/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -n 1)"
    if [ -n "$want" ] && [ "$venv_version" != "$want" ]; then
      warn "the venv toolkit is CUDA $venv_version but torch was built with $want"
      if [ "$check_only" -eq 0 ]; then
        echo "installing the matching toolkit"
        install_cuda "$want" && cuda_ok=1
      fi
    else
      CUDA_HOME="$venv_cuda"
      export CUDA_HOME
      export PATH="$CUDA_HOME/bin:$PATH"
      cuda_ok=1
    fi
  elif [ -z "$want" ]; then
    warn "torch reports no CUDA build, so the toolkit is skipped."
  elif install_cuda "$want"; then
    cuda_ok=1
  fi
fi

if [ "$fast" -eq 1 ]; then
  echo "fast mode, skipping the optional causal_conv1d kernel"
  echo "the LFM path uses the reference convolution, which is slower per step"
elif [ "$cuda_ok" -eq 1 ]; then
  if ! command -v g++ >/dev/null 2>&1; then
    warn "g++ is missing, so no CUDA extension can compile."
    echo "  sudo apt-get install -y build-essential"
  else
    echo "installing causal_conv1d"
    # Build isolation would pull a second torch into a temporary environment.
    # Build against the venv torch so the extension matches the runtime. CC is
    # what torch hands to nvcc as the host compiler, and a plain gcc of a
    # different major version has no C++ front end.
    if ! env CC=g++ CXX=g++ "$venv/bin/python" -m pip install -q --no-build-isolation causal_conv1d; then
      warn "causal_conv1d did not install, the LFM path uses the slow reference kernel."
    fi
  fi
else
  echo "skipping causal_conv1d, no CUDA toolkit to build it"
fi

echo "versions"
"$venv/bin/python" - <<'PY'
import torch
import transformers
import peft
import accelerate
import vllm

print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"peft         {peft.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"vllm         {vllm.__version__}")
print(f"cuda         {torch.cuda.is_available()}")
PY

echo "done"
echo
echo "READY. start training with:"
echo "  ./run.sh code-score --config configs/lfm-spread.toml --runs-root runs"
echo "run.sh sets CUDA_HOME from the venv, so the fast kernels are used."
echo "the dashboard reads the same runs folder:"
echo "  $venv/bin/rlvr dashboard --runs-root runs --port 8081"
