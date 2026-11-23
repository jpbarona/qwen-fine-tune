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
# The script probes the forwarded URL from the box before it reports success.
# The tunnel is only usable when that probe answers, so a forward that did not
# bind is a failure, not a silent success.
#
# The tunnel runs detached and reconnects when the link drops. Its pid goes in
# ~/.cache/rlvr/tunnel-<host>-<port>.pid, and the gateway and port it bound go
# in the matching .state file for the launcher to read. Run this again to reuse
# a tunnel that is already open. One tunnel serves every arm on the box.
#
# Environment:
#   SSH_KEY                  private key, default ~/.ssh/id_ed25519
#   RLVR_TUNNEL_DIR          pid and log directory, default ~/.cache/rlvr
#   RLVR_TUNNEL_RETRY_DELAY  seconds between reconnect attempts, default 5
#   RLVR_PROBE_ATTEMPTS      forward probes before giving up, default 30
#   RLVR_PROBE_INTERVAL      seconds between probes, default 2

set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: open-tunnel.sh <host> [port]" >&2
  exit 2
fi

host="$1"
port="${2:-8081}"
key_path="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
cache_dir="${RLVR_TUNNEL_DIR:-$HOME/.cache/rlvr}"
retry="${RLVR_TUNNEL_RETRY_DELAY:-5}"
probe_attempts="${RLVR_PROBE_ATTEMPTS:-30}"
probe_interval="${RLVR_PROBE_INTERVAL:-2}"

# The box is reached as Ubuntu@<host>. A caller that already supplied a user
# keeps it. The cache key drops the user, so Ubuntu@box and box share a tunnel.
# The server binds the tailnet address on this VM, not loopback. Read the
# address from the interface so a changed tailnet address needs no edit here.
bind_host="${MLFLOW_BIND_HOST:-$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
bind_host="${bind_host:-127.0.0.1}"

target="$host"
case "$target" in
  *@*) ;;
  *) target="Ubuntu@$host" ;;
esac
bare_host="${host##*@}"
safe_host="${bare_host//[^A-Za-z0-9_.-]/_}"
tunnel_key="$safe_host-$port"

mkdir -p "$cache_dir"
pid_file="$cache_dir/tunnel-$tunnel_key.pid"
running_file="$cache_dir/tunnel-$tunnel_key.running"
state_file="$cache_dir/tunnel-$tunnel_key.state"
log_file="$cache_dir/tunnel-$tunnel_key.log"

# A pid file is only trusted when its process is alive, the state file it
# should have written is present, and the forward still answers. Anything else
# is a leftover or a dead link.
probe_forward() {
  local url="$1"
  local probe=1
  while [ "$probe" -le "$probe_attempts" ]; do
    if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key_path" \
        "$target" "curl -sSf $url" >/dev/null 2>&1; then
      return 0
    fi
    probe=$((probe + 1))
    if [ "$probe" -le "$probe_attempts" ]; then
      sleep "$probe_interval"
    fi
  done
  return 1
}

if [ -s "$pid_file" ]; then
  existing="$(cat "$pid_file")"
  if kill -0 "$existing" 2>/dev/null && [ -s "$state_file" ]; then
    # shellcheck disable=SC1090
    . "$state_file"
    if probe_forward "http://$gateway:$port/health"; then
      echo "tunnel to $bare_host is already open (pid $existing)"
      exit 0
    fi
    echo "open-tunnel.sh: the open tunnel to $bare_host does not answer" \
      "http://$gateway:$port/health, see $log_file" >&2
    exit 1
  fi
fi

# Unlink the stale files before starting a new loop. Without this a stale pid
# file satisfies the wait below and the script reports a tunnel that is not
# there.
if [ -s "$pid_file" ]; then
  existing="$(cat "$pid_file")"
  if tr '\0' ' ' <"/proc/$existing/cmdline" 2>/dev/null | grep -q tunnel_loop; then
    kill -TERM -- "-$existing" 2>/dev/null || kill -TERM "$existing" 2>/dev/null
  fi
fi
rm -f "$pid_file" "$running_file" "$state_file"

gateway="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key_path" "$target" \
  "docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'" \
  2>/dev/null | tr -d '[:space:]')"
if [ -z "$gateway" ]; then
  gateway="172.17.0.1"
  echo "open-tunnel.sh: the box reported no gateway, trying $gateway"
fi

cleanup_files() {
  rm -f "$pid_file" "$running_file" "$state_file"
}

tunnel_loop() {
  current_child=""
  cleanup() {
    if [ -n "$current_child" ]; then
      kill -TERM "$current_child" 2>/dev/null
    fi
    cleanup_files
  }
  # A fatal signal is handled where it arrives. An EXIT trap alone is not
  # enough, because bash does not run it when SIGTERM ends the shell.
  trap cleanup EXIT
  trap 'cleanup; exit 1' INT TERM

  printf 'pid=%s\nstarted=%s\n' "$$" "$(date +%s%N)" >"$running_file"
  while true; do
    # Keep the log bounded. Each write appends, so trimming the file here
    # cannot leave a hole in it.
    if [ -f "$log_file" ] && [ "$(wc -c <"$log_file")" -gt 1048576 ]; then
      tail -n 500 "$log_file" >"$log_file.trim" 2>/dev/null &&
        mv "$log_file.trim" "$log_file"
    fi
    ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$key_path" -N \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -R "$gateway:$port:$bind_host:$port" "$target" >>"$log_file" 2>&1 &
    current_child=$!
    wait "$current_child"
    current_child=""
    sleep "$retry" &
    current_child=$!
    wait "$current_child"
    current_child=""
  done
}

export -f tunnel_loop
export -f cleanup_files
export running_file pid_file state_file log_file gateway port bind_host target retry key_path

start_ns="$(date +%s%N)"
setsid bash -c tunnel_loop >/dev/null 2>&1 &
launcher_pid=$!

# Wait for the pid this launch wrote, not any pid. A pid older than the start
# of this launch is stale and is ignored.
loop_pid=""
n=0
while [ "$n" -lt 100 ]; do
  if [ -s "$running_file" ]; then
    pid=""
    started=0
    # shellcheck disable=SC1090
    . "$running_file"
    if [ -n "$pid" ] && [ "$started" -ge "$start_ns" ] && kill -0 "$pid" 2>/dev/null; then
      loop_pid="$pid"
      break
    fi
  fi
  n=$((n + 1))
  sleep 0.05
done

on_signal() {
  echo "open-tunnel.sh: interrupted, closing the tunnel" >&2
  if [ -n "$loop_pid" ]; then
    kill -TERM -- "-$loop_pid" 2>/dev/null || kill -TERM "$loop_pid" 2>/dev/null
  else
    kill -TERM -- "-$launcher_pid" 2>/dev/null || kill -TERM "$launcher_pid" 2>/dev/null
  fi
  cleanup_files
  exit 1
}
trap on_signal INT TERM

if [ -z "$loop_pid" ]; then
  echo "open-tunnel.sh: the tunnel did not start, see $log_file" >&2
  kill -TERM -- "-$launcher_pid" 2>/dev/null || true
  cleanup_files
  exit 1
fi

# Probe the forwarded URL from the box. This is the only check that proves the
# forward bound on the container gateway, because it goes through sshd.
health_url="http://$gateway:$port/health"
if ! probe_forward "$health_url"; then
  echo "open-tunnel.sh: no answer from $health_url through $bare_host after" \
    "$probe_attempts probes" >&2
  echo "open-tunnel.sh: check GatewayPorts and the MLflow server, see $log_file" >&2
  kill -TERM -- "-$loop_pid" 2>/dev/null || kill -TERM "$loop_pid" 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$loop_pid" 2>/dev/null || break
    sleep 0.1
  done
  cleanup_files
  exit 1
fi

# Publish the pid and the bound address only now that the probe passed.
printf 'gateway=%s\nport=%s\n' "$gateway" "$port" >"$state_file"
printf '%s\n' "$loop_pid" >"$pid_file"

echo "tunnel to $bare_host forwards $gateway:$port to $bind_host:$port (pid $loop_pid)"
