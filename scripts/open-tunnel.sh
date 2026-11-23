#!/usr/bin/env bash
# Open a reverse tunnel so the box can reach the MLflow server on this VM.
#
# Usage:
#   ./scripts/open-tunnel.sh <host> [port]
#
# The forward binds the Docker bridge gateway on the box, so the training
# container reaches the server at http://<gateway>:<port>. The bind address is
# read on the box, which removes the guess on a box with a non-default bridge.
# `GatewayPorts clientspecified` must be set in the box `sshd_config`, otherwise
# sshd refuses a bind on anything but loopback.
#
# The tunnel runs detached and reconnects when the link drops. Its pid goes in
# ~/.cache/rlvr/tunnel-<host>.pid. Run this again to reuse a tunnel that is
# already open. One tunnel serves every arm on the box.
#
# Environment:
#   SSH_KEY                  private key, default ~/.ssh/id_ed25519
#   RLVR_TUNNEL_DIR          pid and log directory, default ~/.cache/rlvr
#   RLVR_TUNNEL_RETRY_DELAY  seconds between reconnect attempts, default 5

set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: open-tunnel.sh <host> [port]" >&2
  exit 2
fi

host="$1"
port="${2:-8081}"
key="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
cache_dir="${RLVR_TUNNEL_DIR:-$HOME/.cache/rlvr}"
retry="${RLVR_TUNNEL_RETRY_DELAY:-5}"

# The box is reached as Ubuntu@<host>. A caller that already supplied a user
# keeps it.
target="$host"
case "$target" in
  *@*) ;;
  *) target="Ubuntu@$host" ;;
esac

safe_host="${host//[^A-Za-z0-9_.-]/_}"
mkdir -p "$cache_dir"
pid_file="$cache_dir/tunnel-$safe_host.pid"
log_file="$cache_dir/tunnel-$safe_host.log"

if [ -s "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "tunnel to $host is already open (pid $(cat "$pid_file"))"
  exit 0
fi

gateway="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key" "$target" \
  "docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'" \
  2>/dev/null | tr -d '[:space:]')"
if [ -z "$gateway" ]; then
  gateway="172.17.0.1"
fi

tunnel_loop() {
  printf '%s\n' "$$" > "$pid_file"
  trap 'rm -f "$pid_file"' EXIT
  while true; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key" -N \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -R "$gateway:$port:127.0.0.1:$port" "$target"
    sleep "$retry"
  done
}

export pid_file gateway port target retry key
export -f tunnel_loop

setsid bash -c tunnel_loop >"$log_file" 2>&1 &

# Wait for the loop to publish its pid. Without one the tunnel did not start.
n=0
until [ -s "$pid_file" ] || [ "$n" -ge 40 ]; do
  sleep 0.05
  n=$((n + 1))
done
if [ ! -s "$pid_file" ]; then
  echo "open-tunnel.sh: the tunnel did not start, see $log_file" >&2
  exit 1
fi

echo "tunnel to $host forwards $gateway:$port to 127.0.0.1:$port (pid $(cat "$pid_file"))"
