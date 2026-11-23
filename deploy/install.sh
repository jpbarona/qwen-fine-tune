#!/usr/bin/env bash
# Install the MLflow server unit and the nightly backup timer.
#
# Run as root for the full install. Without root the script prepares the
# store directory and the environment file, prints the two privileged
# commands, and exits 0. A real failure exits nonzero.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEPLOY_DIR="$REPO_DIR/deploy"
UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"
DATA_DIR="${RLVR_MLFLOW_DATA_DIR:-/home/opencode/Work/rl-grpo-data/mlflow}"
ENV_DIR="${MLFLOW_CONFIG_DIR:-/home/opencode/.config/mlflow}"
ENV_FILE="$ENV_DIR/env"
MLFLOW_BIN="$REPO_DIR/.venv/bin/mlflow"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
UNITS=(mlflow.service mlflow-backup.service mlflow-backup.timer)

if [ ! -x "$MLFLOW_BIN" ]; then
  echo "missing $MLFLOW_BIN" >&2
  echo "install the tracking extra into the repo venv first" >&2
  exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "missing $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$ENV_DIR"

if [ ! -e "$ENV_FILE" ]; then
  install -m 0600 "$DEPLOY_DIR/mlflow.env.example" "$ENV_FILE"
  echo "wrote $ENV_FILE from the example, fill in the key values" >&2
fi

if [ "$(id -u)" -eq 0 ]; then
  for unit in "${UNITS[@]}"; do
    install -m 0644 "$DEPLOY_DIR/$unit" "$UNIT_DIR/$unit"
  done
  chown opencode:opencode "$DATA_DIR"
  chown -R opencode:opencode "$ENV_DIR"
  systemctl daemon-reload
  systemctl enable mlflow.service
  systemctl enable --now mlflow-backup.timer
  systemctl restart mlflow.service
  systemctl list-timers mlflow-backup.timer --no-pager
  echo "installed $DATA_DIR and the units in $UNIT_DIR"
else
  unit_args=""
  for unit in "${UNITS[@]}"; do
    unit_args="$unit_args $DEPLOY_DIR/$unit"
  done
  echo "run these two commands as root to finish:"
  echo "  sudo install -m 0644$unit_args $UNIT_DIR"
  echo "  sudo systemctl daemon-reload && sudo systemctl enable --now mlflow.service mlflow-backup.timer"
fi
