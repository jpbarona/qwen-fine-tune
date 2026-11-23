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

## Dashboard

The view is MLflow. The server binds `0.0.0.0:8081`. Tunnel it from this VM:

```bash
ssh -N -L 100.67.80.52:8083:127.0.0.1:8081 -i ~/.ssh/id_ed25519 Ubuntu@<host>
```

The tunnel points at `127.0.0.1:8081` on the host. Then open
`http://100.67.80.52:8083/`.

The store and artifacts live under the runs root. The image serves the same
server with `./scripts/docker-run.sh dashboard`. The store starts empty. Import
an old run folder once with `rlvr import-mlflow --runs-root runs`.

## Before the box is destroyed

Copy the run folders off. They exist only on the box.

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
