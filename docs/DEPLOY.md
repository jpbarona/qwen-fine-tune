# Deployment

One rented GPU box. Massed Compute, RTX A6000 with 48 GB, 96 GB RAM, 300 GB
disk, 0.57 dollars an hour.

## Access

- Host: read it from the provider panel. It changes per instance.
- User: `Ubuntu`
- Key: `/home/opencode/.ssh/id_ed25519`, comment `opencode-agent`
- Login: `ssh -i ~/.ssh/id_ed25519 Ubuntu@<host>`

The instance trusts the `opencode-agent` key and the Mac key. No password is
stored in this repository.

A fresh box gets a new SSH host key. Remove the old entry before connecting:

```bash
ssh-keygen -f ~/.ssh/known_hosts -R <host>
```

## First boot

```bash
scp -i ~/.ssh/id_ed25519 -r docker scripts configs \
  Ubuntu@<host>:~/rl-grpo/
ssh -i ~/.ssh/id_ed25519 Ubuntu@<host>
cd ~/rl-grpo
GHCR_TOKEN=... ./scripts/docker-setup.sh
```

The box does not clone the repository. It is private and the box has no key, so
copy the tree from this VM. `scp` and `rsync` both work.

The screen file is not in git either. Restore it once:

```bash
rclone copy hetzner-dumps:opencodeagent1/dumps/screens/2026-11-21-lfm-base \
  screens/mbpp-base
```

## Keys

`OPENROUTER_API_KEY` is read from the environment. Keep it in `~/.env` on the box.

```bash
set -a && . ~/.env && set +a
```

`OPENCODE_GO_API_KEY` is read only when `[judge] provider = "opencode"`.

The code arm needs no judge key. The hybrid and gated arms do.

## Run

```bash
cd ~/rl-grpo
set -a && . ~/.env && set +a
./scripts/docker-run.sh code-score --config configs/lfm-spread.toml --runs-root runs
```

Detached, with a watcher from this VM:

```bash
ssh -i ~/.ssh/id_ed25519 Ubuntu@<host> \
  'cd ~/rl-grpo && setsid nohup ./scripts/docker-run.sh code-score \
   --config configs/lfm-spread.toml --runs-root runs \
   > ~/run.log 2>&1 < /dev/null &'
./scripts/watch-remote.sh Ubuntu@<host> "rlv[r] run" 60
```

`watch-remote.sh` reports on exit, treats a failed SSH as a retry rather than
completion, and gives up after eight empty replies.

A run that should write to the store on this VM goes through
`./scripts/start-remote-run.sh` instead. The Dashboard section covers it.

## Dashboard

The store lives on this VM, not on the box. One MLflow server serves every box,
so a run keeps its history after the box is destroyed.

The server is a systemd unit in `deploy/`, run as `opencode`. Once installed it
binds `127.0.0.1:8081`, keeps its SQLite store at
`/home/opencode/Work/rl-grpo-data/mlflow/mlflow.db`, and proxies artifacts to
`s3://opencodeagent1/mlflow/v1/artifacts`. The credentials are in
`/home/opencode/.config/mlflow/env`. Install the unit and the backup timer with:

```bash
sudo deploy/install.sh
```

`deploy/README.md` covers the install, the health check and the restore. The
install needs sudo, which this VM does not allow. See [ENVIRONMENT.md](ENVIRONMENT.md) for
how the server actually runs here.

The box reaches the server through an SSH reverse tunnel opened by this VM. The
tunnel binds the Docker bridge gateway on the box, so the training container
reaches it without a public port. It needs `GatewayPorts clientspecified` in the
box `sshd_config`. `scripts/docker-setup.sh` writes that line as a drop-in and
reloads sshd.

Open the tunnel alone:

```bash
./scripts/open-tunnel.sh Ubuntu@<host>
```

Open the tunnel and start a run:

```bash
./scripts/start-remote-run.sh code-score Ubuntu@<host> \
  --config configs/lfm-spread.toml --runs-root runs
```

The script opens the tunnel, reads the gateway the tunnel bound, waits for
`/health` from the box, then starts `docker-run.sh` detached. It launches nothing
when the health check does not answer, because that run would train with no
store. The run log is `$HOME/.cache/rlvr/run-<arm>-<host>.log`.

The tunnel reconnects when the link drops. One tunnel serves every arm on the
box. Run `open-tunnel.sh` again to reuse an open tunnel.

`rlvr dashboard` prints the store URL:

```bash
.venv/bin/rlvr dashboard
```

Open the UI through the Mac to VM tunnel on port 8081. That tunnel forwards
`8081` on the Mac to `127.0.0.1:8081` on this VM. Then open
`http://127.0.0.1:8081/`.

`rlvr dashboard --local` starts the old local server on port 8099 with the store
under the runs root. It starts a server from the local venv and needs no tunnel.

A run with `[mlflow] bucket` set copies its checkpoints and run folder to
`s3://opencodeagent1/runs/v1/<arm>/<started>/` while it trains. The uploader
reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `MLFLOW_S3_ENDPOINT_URL`
from the box environment, so keep them in `$HOME/.env`. The key is scoped to the
`runs/` prefix. The run folder stays the record and the bucket is the copy.

After a dropped link, replay one run folder into the store:

```bash
.venv/bin/rlvr sync-mlflow runs/code-score/<timestamp>
```

The command deletes and rebuilds that run's entries, so it is safe to repeat. It
matches the run on `rlvr/run_name`, the arm and start timestamp, not the folder
path.

`deploy/mlflow-backup.timer` copies the store nightly at 03:00 UTC and keeps 14
copies under `s3://opencodeagent1/mlflow/v1/backups/`. The store is rebuildable
from the run folders, so the backup saves that work.

The store starts empty. Old dumps under `dumps/` are untouched. Import an old
run folder once with `rlvr import-mlflow --runs-root runs`.

## Before the box is destroyed

A run with `[mlflow] bucket` set copies its folder to the bucket as it trains.
Every regular file in the run folder is copied, `gpu.jsonl`, `rollouts.jsonl`
and `mlflow.json` included, and `manifest.json` is written last. The manifest
is not in the local folder, so exclude it when you compare:

```bash
rclone check <run folder> \
  hetzner-dumps:opencodeagent1/runs/v1/<arm>/<started> --exclude manifest.json
```

A run without a bucket exists only on the box. Copy it off:

```bash
rclone copy <run folder> hetzner-dumps:opencodeagent1/dumps/runs/gpu/<timestamp> --progress
rclone check <run folder> hetzner-dumps:opencodeagent1/dumps/runs/gpu/<timestamp>
```

Delete the local copy only after the check reports zero differences. Update
`/home/opencode/Work/rl-grpo-data-INDEX.md`.

## Costs

Idle time bills at the same rate. Release the box when no run is active.

| Event | Cost |
| --- | --- |
| One hour | 0.57 |
| Fresh setup, container | about 0.05 |
| One hundred steps | about 2.60 |
