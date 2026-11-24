#!/usr/bin/env bash
# Keep the MLflow server alive on the tailnet.
#
# The browser loads the UI from this origin, so the same origin must be allowed
# for CORS. Without it every POST from the UI returns 403 and the runs table
# shows INTERNAL_ERROR.
#
# systemd is not usable on this VM, so run this in a tmux session:
#   tmux new-session -d -s mlflow /home/opencode/Work/rl-grpo/deploy/mlflow-serve.sh

set -euo pipefail

mlflow_home="${MLFLOW_HOME:-/home/opencode/Work/rl-grpo-data/mlflow}"
python="${MLFLOW_PYTHON:-/home/opencode/Work/rl-grpo/.venv/bin/python}"
host="${MLFLOW_HOST:-100.67.80.52}"
port="${MLFLOW_PORT:-8081}"
origins="${MLFLOW_CORS_ORIGINS:-http://$host:$port}"

set -a
# shellcheck disable=SC1090
. "${MLFLOW_ENV:-/home/opencode/.config/mlflow/env}"
set +a

while true; do
  "$python" -m mlflow server \
    --backend-store-uri "sqlite:///$mlflow_home/mlflow.db" \
    --artifacts-destination s3://opencodeagent1/mlflow/v1/artifacts \
    --host "$host" \
    --port "$port" \
    --workers 1 \
    --allowed-hosts '*' \
    --cors-allowed-origins "$origins" \
    >> "$mlflow_home/mlflow.log" 2>&1
  sleep 5
done
