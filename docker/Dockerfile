# Training image for the rlvr study.
#
# The base carries CUDA 13.0 and nvcc. That version matches the torch wheel pin,
# so the CUDA extension and the vLLM kernels compile without a version fight.
#
# The repository is not baked in. It is mounted at /app at run time, so a code
# change needs no rebuild. Only a dependency change does.

FROM nvidia/cuda:13.0.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common curl git ca-certificates \
        build-essential ninja-build \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer. It is cached until pyproject.toml changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -e ".[vllm]" \
    && /opt/venv/bin/pip install --no-cache-dir "pytest>=9,<10"

# The optional convolution kernel, built for the study GPU only.
ARG CUDA_ARCH=8.6
RUN TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} CC=g++ CXX=g++ \
    /opt/venv/bin/pip install --no-build-isolation --no-cache-dir causal_conv1d

# The test runner shells out to `docker` and talks to the mounted host socket.
# Without the client every rollout records runner_error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
CMD ["bash"]
