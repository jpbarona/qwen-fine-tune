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
exec "$rlvr_venv/bin/rlvr" run --arm "$arm" "$@"
