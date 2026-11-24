#!/usr/bin/env bash
# Copy this checkout to a fresh GPU box.
#
# Usage:
#   ./scripts/sync-to-box.sh <host> [user]
#
# The box gets ~/rl-grpo with its git history, so the run records the commit it
# trained. Secrets are not copied. Copy ~/.ghcr-token and ~/.env to the box once
# per instance, as DEPLOY.md says. The screen folder is copied when it exists
# here, because it is not tracked by git.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: sync-to-box.sh <host> [user]" >&2
  exit 2
fi

host="$1"
user="${2:-Ubuntu}"
case "$host" in
  *@*) target="$host" ;;
  *) target="$user@$host" ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$repo_root/.git" ]; then
  echo "sync-to-box.sh: this checkout is a worktree, so its .git is a pointer" >&2
  echo "sync-to-box.sh: run this from a full clone, or the box cannot tag the commit" >&2
  exit 2
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" true; then
  echo "sync-to-box.sh: cannot ssh to $target" >&2
  exit 1
fi

ssh -o BatchMode=yes "$target" 'mkdir -p ~/rl-grpo'

rsync -a --delete \
  --exclude .venv \
  --exclude runs \
  --exclude screens \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude .ruff_cache \
  --exclude .runner-tmp \
  --exclude judge-cache \
  -e "ssh -o BatchMode=yes" \
  "$repo_root/" "$target:rl-grpo/"

if [ -d "$repo_root/screens" ]; then
  rsync -a -e "ssh -o BatchMode=yes" \
    "$repo_root/screens/" "$target:rl-grpo/screens/"
else
  echo "sync-to-box.sh: no screens/ here. the box needs screens/mbpp-base/rows.jsonl" >&2
  exit 1
fi

echo "synced $(git -C "$repo_root" rev-parse --short HEAD) to $target:~/rl-grpo"
