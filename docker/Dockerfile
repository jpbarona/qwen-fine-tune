# syntax=docker/dockerfile:1
# Training image for the rlvr study.
#
# The base carries CUDA 13.0 and nvcc. That version matches the torch wheel pin,
# so the CUDA extension and the vLLM kernels compile without a version fight.
#
# The repository is not baked in. It is mounted at /app at run time, so a code
# change needs no rebuild. Only a dependency change does.
#
# Build with scripts/build-image.sh. It tags the image and pushes it, so a fresh
# box pulls instead of building. The build is only paid when pyproject.toml or
# this file changes.

FROM nvidia/cuda:13.0.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy

# The apt cache mounts keep downloaded packages between builds. They are not
# part of the image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common curl git ca-certificates \
        build-essential ninja-build \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev

WORKDIR /app

# uv installs the Python stack in a fraction of pip's time. Its cache mount
# survives between builds, so a dependency change reuses the wheels.
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir uv

# The source is copied before the install because an editable install needs it.
# The uv cache makes the reinstall cheap when only code changed.
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    /opt/venv/bin/uv pip install --python /opt/venv/bin/python -e ".[vllm,tracking]" \
    && /opt/venv/bin/uv pip install --python /opt/venv/bin/python \
        "psutil" "nvidia-ml-py" "pytest>=9,<10"

# The optional convolution kernel, built for the study GPU only.
ARG CUDA_ARCH=8.6
RUN --mount=type=cache,target=/root/.cache/uv \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} CC=g++ CXX=g++ \
    /opt/venv/bin/uv pip install --python /opt/venv/bin/python \
        --no-build-isolation causal_conv1d

# triton 3.8.0 compiles its C driver with the C++ compiler when CC is g++.
# The CUDA 13 headers reject the int to enum conversions that follow, so the
# first vLLM engine start fails. Compile the C source with gcc instead.
RUN /opt/venv/bin/python - <<'PY'
import pathlib

path = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/triton/runtime/build.py"
)
text = path.read_text()
old = "    cc = _find_compiler(language)\n"
new = '    cc = "gcc" if language == "c" else _find_compiler(language)\n'
if old in text:
    path.write_text(text.replace(old, new))
PY

# The test runner shells out to `docker` and talks to the mounted host socket.
# Without the client every rollout records runner_error.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends docker.io

WORKDIR /app
CMD ["bash"]
