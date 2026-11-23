# Agent notes

Read `README.md` first for what the study is and how a run works. `docs/INDEX.md`
points at every design, plan and report. This file covers the things an agent
gets wrong once and then repeats.

## Tools on this VM

`~/.local/bin` is not always on `PATH` in a non-interactive shell. Call these
by full path.

| What | Where | Notes |
| --- | --- | --- |
| GitHub CLI | `~/.local/bin/gh` | Authenticated as `jpbarona`. Use it to open a pull request. |
| GitHub token | `~/.github-token` | Fine-grained, mode 600. Never print it or copy it into a file. |
| GHCR token | `~/.github-token-packages` | Pulls the private training image. `GHCR_USER` and `GHCR_TOKEN` for `scripts/docker-setup.sh`. |
| uv | `~/.local/bin/uv` | Installed on the VM, not pinned by this repository yet. |
| rclone | `~/.config/rclone/rclone.conf` | Remote `hetzner-dumps`, bucket `opencodeagent1`. A root shell needs `--config` with this path. |
| MLflow credentials | `~/.config/mlflow/env` | S3 key pair for the artifact proxy and the backup. Mode 600. |

Open a pull request with:

```bash
~/.local/bin/gh pr create --base main --head <branch> --title "<title>" --body-file <file>
```

## Layout rules

The package is layered. `.importlinter` enforces the direction and CI runs it.

```text
app -> report -> tracking -> training -> engine | evaluation -> core
```

`rlvr.training.experiment` is the only orchestration point. Seam values live in
`rlvr.core.values`. Shared constants live in `rlvr.core.constants`. Do not
import a seam value from `experiment`, and do not add an upward import.

## Commands

```bash
.venv/bin/lint-imports                  # layer contract
.venv/bin/python -m pytest -q           # full suite, about five minutes
.venv/bin/python -m pytest -q tests/core
.venv/bin/rlvr run --arm code-score --fake --config configs/smoke.toml --runs-root /tmp/runs
```

## Test caveat on this VM

`configs/real.toml` names a bucket, and a fake run with that config starts the
uploader. A test shell has no bucket credentials, so the uploader retries and
the run hangs at exit. Run the suite with a short close timeout and a dead
endpoint:

```bash
RLVR_UPLOAD_CLOSE_SECONDS=1 MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9 \
  AWS_EC2_METADATA_DISABLED=true AWS_ACCESS_KEY_ID=x AWS_SECRET_ACCESS_KEY=y \
  .venv/bin/python -m pytest -q
```

This is a local workaround, not the intended production setting.
