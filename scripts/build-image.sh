#!/usr/bin/env bash
# Build and push the training image.
#
# Run this on a machine with Docker and the ghcr token. The GPU box is fine.
# Only run it when pyproject.toml or docker/Dockerfile changed. A code change
# needs no rebuild, because the repository is mounted at run time.
#
# Usage:
#   GHCR_TOKEN=... ./scripts/build-image.sh [tag]
#
# The tag defaults to the UTC timestamp. The script also moves `latest`, prints
# the digest, and leaves it in docker/last-digest.txt.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

image="${RLVR_IMAGE_NAME:-ghcr.io/jpbarona/rlvr}"
tag="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
user="${GHCR_USER:-jpbarona}"

if [ -z "${GHCR_TOKEN:-}" ]; then
  echo "build-image.sh: GHCR_TOKEN is not set" >&2
  exit 2
fi

echo "$GHCR_TOKEN" | docker login ghcr.io -u "$user" --password-stdin

docker build -f docker/Dockerfile -t "$image:$tag" -t "$image:latest" .
docker push "$image:$tag"
docker push "$image:latest"

digest="$(docker image inspect "$image:$tag" --format '{{index .RepoDigests 0}}')"
mkdir -p docker
printf '%s\n' "$digest" > docker/last-digest.txt
printf '%s\n' "$digest"
