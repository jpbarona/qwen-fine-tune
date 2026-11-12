#!/usr/bin/env bash
# Launch one reward arm with no manual edits.
#
# Usage:
#   ./run.sh <code-score|hybrid|gated-judge|gated-process> [-- extra args]
#
# The judge key is read from the OPENCODE_GO_API_KEY environment variable by the
# judge client. It is never written to this file, a config, or a run folder.
# Extra arguments are passed through to `rlvr run`, so a smoke run is:
#
#   ./run.sh code-score --config configs/smoke.toml
#
# Resume from the last checkpoint after a crash:
#
#   ./run.sh code-score --resume

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: run.sh <code-score|hybrid|gated-judge|gated-process> [-- extra args]" >&2
  exit 2
fi

arm="$1"
shift

case "$arm" in
  code-score|hybrid|gated-judge|gated-process) ;;
  *)
    echo "run.sh: unknown arm '$arm'" >&2
    exit 2
    ;;
esac

exec .venv/bin/rlvr run --arm "$arm" --config configs/real.toml "$@"
