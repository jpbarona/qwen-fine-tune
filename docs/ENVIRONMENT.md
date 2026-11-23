# This VM

What is true on the VM that runs the study. The README explains the study. This
file explains the machine. Every fact here has its only home here.

## From the Mac

```bash
ssh opencode@2.31.62.7
```

Forward the MLflow UI to the Mac. The server binds loopback only, so the tunnel
is the way in.

```bash
ssh -N -L 8081:127.0.0.1:8081 opencode@2.31.62.7
```

Then open `http://127.0.0.1:8081`.

## No sudo

`sudo` fails with `The "no new privileges" flag is set`. There is no systemd
user bus and the user crontab is unreadable. `deploy/install.sh` cannot run on
this VM. The units in `deploy/` apply on a host where sudo works.

## Tools

`~/.local/bin` is not always on `PATH` in a non-interactive shell. Call these by
full path.

| What | Where | Notes |
| --- | --- | --- |
| GitHub CLI | `~/.local/bin/gh` | Authenticated as `jpbarona`. `gh pr create` works. |
| GitHub token | `~/.github-token` | Fine-grained, mode 600. |
| GHCR token | `~/.github-token-packages` | Pulls the private training image. Used as `GHCR_TOKEN` with `GHCR_USER`. |
| uv | `~/.local/bin/uv` | Installed on this VM. The repository still installs with pip. |
| rclone | `~/.config/rclone/rclone.conf` | Remote `hetzner-dumps`, bucket `opencodeagent1`. A root shell needs `--config` with this path. |
| MLflow credentials | `~/.config/mlflow/env` | S3 key pair for the artifact proxy and the backup. Mode 600. |

Never print a token or copy one into a tracked file.

## MLflow

- Server: `127.0.0.1:8081`, one worker, SQLite store.
- Store: `/home/opencode/Work/rl-grpo-data/mlflow/mlflow.db`.
- Artifacts: `s3://opencodeagent1/mlflow/v1/artifacts`, through the artifact
  proxy. Tested end to end on 23 November 2026.
- Log: `/home/opencode/Work/rl-grpo-data/mlflow/mlflow.log`.
- Supervision: a tmux session named `mlflow` restarts the server five seconds
  after any exit, because systemd cannot be used here.
- Backup: `deploy/backup.py` copies the store to
  `s3://opencodeagent1/mlflow/v1/backups`. Nothing schedules it on this VM. Run
  it by hand:

```bash
set -a && . ~/.config/mlflow/env && set +a
.venv/bin/python deploy/backup.py
```

The store started empty on 23 November 2026. Old runs stay archived in the
bucket under `dumps/` and are not imported.

## Test suite caveat

`configs/real.toml` names a bucket, so a fake run starts the uploader. A test
shell has no bucket credentials, so the uploader retries and the run waits at
exit. Run the suite with a short close timeout and a dead endpoint:

```bash
RLVR_UPLOAD_CLOSE_SECONDS=1 MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9 \
  AWS_EC2_METADATA_DISABLED=true AWS_ACCESS_KEY_ID=x AWS_SECRET_ACCESS_KEY=y \
  .venv/bin/python -m pytest -q
```

This is a local workaround. The production run keeps the real credentials from
the box environment.
