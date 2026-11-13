# Deployment instance

Massed Compute RTX A6000 48GB, Ubuntu Server 24.04.

## Access

- Host: `209.137.198.122`
- User: `Ubuntu`
- Key: `/home/opencode/.ssh/id_ed25519` (`opencode-agent`)
- Login: `ssh -i ~/.ssh/id_ed25519 Ubuntu@209.137.198.122`

The instance accepts the `opencode-agent` key and jpbarona's Mac key. Password login was used once to install those keys. No password is stored in this repo.

## Remote layout

- Project: `~/rl-grpo`
- Virtualenv: `~/rl-grpo/.venv`
- Judge key: already present in `~/.env` as `OPENCODE_GO_API_KEY`
- Docker is installed. The `Ubuntu` user may need the `docker` group.

## Run

```bash
cd ~/rl-grpo
set -a && . ~/.env && set +a
./run.sh code-score --config configs/smoke.toml
./run.sh code-score
```

## Copy the repo again

From this VM:

```bash
rsync -az --exclude .venv --exclude runs --exclude .git \
  /home/opencode/Work/rl-grpo/ Ubuntu@209.137.198.122:~/rl-grpo/
```
