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
- Judge key: `OPENROUTER_API_KEY` is required and read from `~/.env` at startup. The judge uses OpenRouter by default.
- Old key: `OPENCODE_GO_API_KEY` is only read when `[judge] provider = "opencode"`.
- Docker is installed and the `Ubuntu` user is in the `docker` group.

## Every detached job gets a watcher

Start the job and its watcher in the same action. A detached job with no local
waiter is a job nobody hears about, and that has already cost a probe.

```bash
ssh -i ~/.ssh/id_ed25519 Ubuntu@HOST 'setsid nohup ./run.sh code-score > ~/run.log 2>&1 < /dev/null &'
./scripts/watch-remote.sh Ubuntu@HOST "configs/lfm.toml"
```

`watch-remote.sh` reports only on exit, treats a failed ssh as a retry rather
than completion, and gives up after eight empty replies.

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
