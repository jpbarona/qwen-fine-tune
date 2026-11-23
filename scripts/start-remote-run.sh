#!/usr/bin/env bash
# Open the tunnel and launch a run on the box, detached.
#
# Usage:
#   ./scripts/start-remote-run.sh <arm> <host> [extra args]
#
# The script opens the reverse tunnel, waits for the server health endpoint,
# then starts `docker-run.sh` in the background. A run is not launched when the
# health check never answers, because that run would train with no store.
# Extra args pass through to `docker-run.sh`.
#
# Environment:
#   MLFLOW_PORT             forwarded port, default 8081
#   RLVR_HEALTH_ATTEMPTS    tries before giving up, default 30
#   RLVR_HEALTH_INTERVAL    seconds between tries, default 2
#   RLVR_TUNNEL_DIR         pid and log directory, default ~/.cache/rlvr

set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: start-remote-run.sh <arm> <host> [extra args]" >&2
  exit 2
fi

arm="$1"
host="$2"
shift 2
if [ "${1:-}" = "--" ]; then
  shift
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${MLFLOW_PORT:-8081}"
attempts="${RLVR_HEALTH_ATTEMPTS:-30}"
interval="${RLVR_HEALTH_INTERVAL:-2}"
cache_dir="${RLVR_TUNNEL_DIR:-$HOME/.cache/rlvr}"

if ! "$repo_root/scripts/open-tunnel.sh" "$host" "$port"; then
  echo "start-remote-run.sh: the tunnel did not open" >&2
  exit 1
fi

health="http://127.0.0.1:$port/health"
ready=false
try=1
while [ "$try" -le "$attempts" ]; do
  if curl -fsS --max-time 5 "$health" >/dev/null 2>&1; then
    ready=true
    break
  fi
  try=$((try + 1))
  if [ "$try" -le "$attempts" ]; then
    sleep "$interval"
  fi
done

if [ "$ready" != true ]; then
  echo "start-remote-run.sh: no answer from $health after $attempts tries" >&2
  echo "start-remote-run.sh: check the MLflow server on this VM before retrying" >&2
  exit 1
fi

safe_host="${host//[^A-Za-z0-9_.-]/_}"
mkdir -p "$cache_dir" "$repo_root/.runner-tmp"
log_file="$cache_dir/run-$arm-$safe_host.log"

setsid nohup "$repo_root/scripts/docker-run.sh" "$arm" "$@" >"$log_file" 2>&1 &
runner_pid=$!

echo "run launched on $host (pid $runner_pid), log $log_file"
echo "store: $health"
