#!/usr/bin/env bash
# Wait for a remote process to finish, and report when it does.
#
# Usage: watch-remote.sh <host> <pattern> [interval_seconds]
#
# Start this in the same action as any detached remote job. A job that is
# started without a local waiter is a job nobody hears about.
#
# The loop treats a failed ssh as a retry, never as completion. Eight failed
# attempts in a row end the wait with a clear message, so a dead box or a
# dropped network cannot look like a finished run.
set -uo pipefail

host=${1:?host}
pattern=${2:?process pattern}
interval=${3:-30}
key=${SSH_KEY:-$HOME/.ssh/id_ed25519}
tries=0
missing=0

while true; do
  if ! out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key" "$host" \
      "pgrep -f '$pattern' >/dev/null && echo ALIVE || echo GONE" 2>/dev/null); then
    tries=$((tries + 1))
    echo "no reply from $host ($tries)"
    if [ "$tries" -ge 8 ]; then
      echo "GAVE UP: $host unreachable"
      exit 1
    fi
    sleep "$interval"
    continue
  fi
  tries=0
  if [ "$out" = "GONE" ]; then
    missing=$((missing + 1))
    # Two consecutive misses, so one scheduling hiccup cannot end the wait.
    if [ "$missing" -ge 2 ]; then
      echo "FINISHED: no process matching $pattern"
      exit 0
    fi
  else
    missing=0
  fi
  sleep "$interval"
done
