#!/usr/bin/env bash
# Open the tunnel and launch a run on the box.
#
# Usage:
#   ./scripts/start-remote-run.sh <arm> <host> [extra args]
#
# The script opens the reverse tunnel, reads the gateway and port that
# open-tunnel.sh actually bound, and checks the health endpoint from the box.
# A run is not launched when that check never answers, because that run would
# train with no store. The launch runs `docker-run.sh` on the box over ssh,
# after sourcing the box environment file. Extra args pass through to
# `docker-run.sh`.
#
# Environment:
#   MLFLOW_PORT             forwarded port, default 8081
#   SSH_KEY                 private key, default ~/.ssh/id_ed25519
#   RLVR_HEALTH_ATTEMPTS    tries before giving up, default 30
#   RLVR_HEALTH_INTERVAL    seconds between tries, default 2
#   RLVR_TUNNEL_DIR         pid, state and log directory, default ~/.cache/rlvr

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
key_path="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

target="$host"
case "$target" in
  *@*) ;;
  *) target="Ubuntu@$host" ;;
esac
bare_host="${host##*@}"
safe_host="${bare_host//[^A-Za-z0-9_.-]/_}"
state_file="$cache_dir/tunnel-$safe_host-$port.state"

if ! "$repo_root/scripts/open-tunnel.sh" "$host" "$port"; then
  echo "start-remote-run.sh: the tunnel did not open, nothing launched" >&2
  exit 1
fi

if [ ! -f "$state_file" ]; then
  echo "start-remote-run.sh: open-tunnel.sh left no state at $state_file" >&2
  exit 1
fi
gateway=""
# shellcheck disable=SC1090
. "$state_file"
if [ -z "$gateway" ] || [ -z "$port" ]; then
  echo "start-remote-run.sh: $state_file does not name a gateway and port" >&2
  exit 1
fi

# The check runs on the box against the forwarded address. A local curl would
# pass with a dead forward, which is the case this gate exists to catch.
health="http://$gateway:$port/health"
ready=false
try=1
while [ "$try" -le "$attempts" ]; do
  if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key_path" \
      "$target" "curl -sSf $health" >/dev/null 2>&1; then
    ready=true
    break
  fi
  try=$((try + 1))
  if [ "$try" -le "$attempts" ]; then
    sleep "$interval"
  fi
done

if [ "$ready" != true ]; then
  echo "start-remote-run.sh: no answer from $health through $bare_host after" \
    "$attempts tries" >&2
  echo "start-remote-run.sh: check the MLflow server and the tunnel, nothing launched" >&2
  exit 1
fi

mkdir -p "$cache_dir"
log_file="$cache_dir/run-$arm-$safe_host.log"

# docker-run.sh passes this to the container with `-e MLFLOW_TRACKING_URI`, so
# the mirror writes to the store on this VM instead of local SQLite. The
# training container runs on the box, so the launch goes over ssh. The box
# environment file holds the bucket keys, which docker-run.sh forwards.
export MLFLOW_TRACKING_URI="http://$gateway:$port"

quoted_args=""
for arg in "$arm" "$@"; do
  quoted_args="$quoted_args$(printf '%q' "$arg") "
done
remote_command="cd ~/rl-grpo && set -a && . ~/.env && set +a && export MLFLOW_TRACKING_URI=$(printf '%q' "$MLFLOW_TRACKING_URI") && ./scripts/docker-run.sh $quoted_args"

ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key_path" \
  "$target" "$remote_command" >"$log_file" 2>&1
status=$?
if [ "$status" -ne 0 ]; then
  echo "start-remote-run.sh: the launch on $bare_host failed with status $status" >&2
  echo "start-remote-run.sh: log $log_file" >&2
  exit "$status"
fi

echo "run launched on $bare_host, log $log_file"
echo "store: $health"
