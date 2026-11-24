#!/usr/bin/env bash
# Prepare a fresh GPU host for training in a container.
#
# Usage:
#   GHCR_TOKEN=... ./scripts/docker-setup.sh
#
# Run once on a new box. It installs Docker and the NVIDIA container runtime if
# they are missing, logs in to the image registry, and pulls the training image
# plus the small Python image the test runner uses.
#
# Root access and docker group membership are separate things. The script asks
# for root directly with `sudo -n` and tests docker separately, so a shell that
# cannot reach the daemon is not mistaken for a shell without root. When the
# caller is not in the docker group the script adds the user, then runs the
# docker commands through `sg docker` for the rest of the run. When it cannot
# do that it stops before any registry call and prints the exact commands to
# run.
#
# The registry login uses two environment variables:
#   GHCR_USER   the GitHub user name, default jpbarona
#   GHCR_TOKEN  a token with read:packages
#
# Without GHCR_TOKEN the login step is skipped, which is correct when the image
# is public.

set -euo pipefail

image="${RLVR_IMAGE:-ghcr.io/jpbarona/rlvr:latest}"
registry="${image%%/*}"
test_image="python:3.12-slim"
toolkit_url="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
gpgkey_url="https://nvidia.github.io/libnvidia-container/gpgkey"
keyring="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

die() {
  printf 'docker-setup.sh: %s\n' "$*" >&2
  exit 1
}

# Root access is its own question. It is needed for apt, the sshd drop-in, the
# daemon, the toolkit and usermod. Docker group membership does not grant it.
if [ "$(id -u)" -eq 0 ]; then
  root_cmd=()
  root_ok=1
elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  root_cmd=(sudo -n)
  root_ok=1
else
  root_cmd=()
  root_ok=0
fi

as_root() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi
  if [ "$root_ok" -eq 0 ]; then
    die "this step needs root: sudo $*. run that, then rerun this script"
  fi
  "${root_cmd[@]}" "$@"
}

# The MLflow server runs on the VM and the box reaches it through an SSH reverse
# tunnel. The tunnel binds the Docker bridge gateway on the box so the training
# container can reach it. sshd refuses a non-loopback bind unless GatewayPorts
# allows it, so a missing line and a `GatewayPorts no` line both need fixing.
# The drop-in is included before the main file on Ubuntu, so it wins. Any other
# GatewayPorts line is rewritten too, because an earlier drop-in would win over
# ours.
ensure_gatewayports() {
  local drop_in=/etc/ssh/sshd_config.d/99-rlvr-gatewayports.conf
  local line="GatewayPorts clientspecified"
  local file

  if [ "$root_ok" -eq 0 ]; then
    echo "warning: no root access, so $drop_in was not written."
    echo "the tunnel will refuse its bind until you run:"
    echo "  echo '$line' | sudo tee $drop_in"
    echo "  sudo sed -i -E 's|^[[:space:]]*GatewayPorts.*|$line|' \\"
    echo "    /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf"
    echo "  sudo sshd -t && sudo systemctl reload ssh"
    return 0
  fi

  if ! as_root mkdir -p /etc/ssh/sshd_config.d; then
    echo "warning: could not create /etc/ssh/sshd_config.d. the tunnel needs:"
    echo "  sudo mkdir -p /etc/ssh/sshd_config.d"
    echo "  echo '$line' | sudo tee $drop_in"
    return 0
  fi

  if ! printf '%s\n' "$line" | as_root tee "$drop_in" >/dev/null; then
    echo "warning: cannot write $drop_in, so the tunnel bind stays refused. run:"
    echo "  echo '$line' | sudo tee $drop_in"
    echo "  sudo sed -i -E 's|^[[:space:]]*GatewayPorts.*|$line|' \\"
    echo "    /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf"
    echo "  sudo sshd -t && sudo systemctl reload ssh"
    return 0
  fi

  for file in /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf; do
    [ -f "$file" ] || continue
    [ "$file" = "$drop_in" ] && continue
    if grep -qE '^[[:space:]]*GatewayPorts([[:space:]]|$)' "$file" 2>/dev/null; then
      if ! as_root sed -i -E \
          "s|^[[:space:]]*GatewayPorts([[:space:]].*)?\$|$line|" "$file"; then
        echo "warning: could not rewrite the GatewayPorts line in $file. run:"
        echo "  sudo sed -i -E 's|^[[:space:]]*GatewayPorts.*|$line|' $file"
        continue
      fi
      echo "replaced the GatewayPorts line in $file"
    fi
  done

  if ! as_root sshd -t; then
    echo "docker-setup.sh: sshd -t rejected the configuration, sshd not reloaded" >&2
    echo "fix /etc/ssh/sshd_config, then run: sudo sshd -t && sudo systemctl reload ssh" >&2
    exit 1
  fi
  if ! as_root systemctl reload ssh 2>/dev/null &&
      ! as_root systemctl reload sshd; then
    echo "warning: the sshd reload failed. run: sudo systemctl reload ssh" >&2
    return 0
  fi
  echo "set $line and reloaded sshd"
}

# A failed `docker info` has several causes. Read the daemon reply before
# deciding what to do, because a permission error and a stopped daemon need
# different fixes.
docker_error() {
  docker info 2>&1 >/dev/null | head -n 1 || true
}

wait_for_docker() {
  local tries=10
  while [ "$tries" -gt 0 ]; do
    if [ "$root_ok" -eq 1 ]; then
      if as_root docker info >/dev/null 2>&1; then
        return 0
      fi
    elif docker info >/dev/null 2>&1; then
      return 0
    else
      # A permission error is not a stopped daemon. Leave it to the access
      # check below instead of waiting for a daemon that is already up.
      case "$(docker_error)" in
        *"permission denied"*) return 1 ;;
      esac
    fi
    tries=$((tries - 1))
    sleep 1
  done
  return 1
}

quote_words() {
  local word out=""
  for word in "$@"; do
    out="$out$(printf '%q' "$word") "
  done
  printf '%s' "$out"
}

# The docker CLI runs as the caller for the registry login, so the credentials
# land in the caller's home. When the caller's session predates the group
# change, `sg docker` supplies the group for one command.
docker_cli() {
  if [ "$docker_access" = sg ]; then
    sg docker -c "$(quote_words docker "$@")"
  else
    docker "$@"
  fi
}

docker_runtime_has_nvidia() {
  local info
  info="$(docker_cli info 2>/dev/null || true)"
  case "$info" in
    *Runtimes:*nvidia*) return 0 ;;
  esac
  return 1
}

ensure_gatewayports

# Docker itself. Installing it needs root. Asking whether the daemon answers,
# below, is a separate question.
if ! command -v docker >/dev/null 2>&1; then
  if [ "$root_ok" -eq 0 ]; then
    die "docker is not installed and this shell has no root access. run:"$'\n'"  sudo apt-get update && sudo apt-get install -y docker.io"$'\n'"then rerun this script"
  fi
  echo "installing docker"
  if ! as_root apt-get update -qq; then
    die "apt-get update failed. run: sudo apt-get update"
  fi
  if ! as_root apt-get install -y -qq docker.io; then
    die "the docker.io install failed. run: sudo apt-get install -y docker.io"
  fi
fi

# Start the daemon through root. This needs no docker group membership, and a
# daemon that is already up is left alone.
if [ "$root_ok" -eq 1 ]; then
  if ! as_root systemctl enable --now docker >/dev/null 2>&1; then
    as_root service docker start >/dev/null 2>&1 || true
  fi
fi

if ! wait_for_docker; then
  err="$(docker_error)"
  case "$err" in
    # The daemon answers but this shell is not allowed to ask it. The access
    # check below adds the group or prints the exact commands.
    *"permission denied"*)
      ;;
    *)
      echo "docker-setup.sh: the docker daemon does not answer." >&2
      echo "docker said: ${err:-no reply}" >&2
      if [ "$root_ok" -eq 1 ]; then
        echo "start it and rerun this script:" >&2
        echo "  sudo systemctl start docker && sudo systemctl status docker" >&2
      else
        echo "this shell has no root access, so the daemon cannot be started here. run:" >&2
        echo "  sudo systemctl start docker" >&2
        echo "  ./scripts/docker-setup.sh" >&2
      fi
      exit 1
      ;;
  esac
fi

# Decide how the caller reaches the daemon. Root and the docker group are
# independent: `docker info` can fail for a shell that has full sudo.
docker_access=plain
if ! docker info >/dev/null 2>&1; then
  user_name="$(id -un)"
  in_group_now=0
  case " $(id -nG) " in
    *" docker "*) in_group_now=1 ;;
  esac

  group_members="$(getent group docker 2>/dev/null | awk -F: '{print $4}' || true)"
  in_group_db=0
  case ",$group_members," in
    *",$user_name,"*) in_group_db=1 ;;
  esac

  if [ "$in_group_now" -eq 1 ]; then
    err="$(docker_error)"
    echo "docker-setup.sh: $user_name is in the docker group but the daemon refused" >&2
    echo "the connection. docker said: ${err:-no reply}" >&2
    echo "check the daemon, then rerun:" >&2
    echo "  sudo systemctl status docker" >&2
    echo "  ./scripts/docker-setup.sh" >&2
    exit 1
  fi

  if [ "$in_group_db" -eq 0 ]; then
    if [ "$root_ok" -eq 0 ]; then
      err="$(docker_error)"
      echo "docker-setup.sh: $user_name is not in the docker group and this shell" >&2
      echo "has no root access, so the group cannot be added here." >&2
      echo "docker said: ${err:-no reply}" >&2
      echo "run these, then rerun this script:" >&2
      echo "  sudo usermod -aG docker $user_name" >&2
      echo "  newgrp docker    # or open a new SSH session" >&2
      echo "  ./scripts/docker-setup.sh" >&2
      exit 1
    fi
    echo "adding $user_name to the docker group"
    if ! as_root usermod -aG docker "$user_name"; then
      die "usermod failed. run: sudo usermod -aG docker $user_name"
    fi
    in_group_db=1
  fi

  # The group is in the database but not in this process. `sg` adds it for one
  # command, so the pulls complete without a new login.
  if ! command -v sg >/dev/null 2>&1; then
    die "$user_name is in the docker group, but this session predates the change and sg is missing. open a new SSH session, then rerun:"$'\n'"  ./scripts/docker-setup.sh"
  fi
  if ! sg_error="$(sg docker -c 'docker info' 2>&1 >/dev/null)"; then
    echo "docker-setup.sh: this session cannot use the docker group yet." >&2
    echo "sg said: ${sg_error:-no reply}" >&2
    echo "run these, then rerun this script:" >&2
    echo "  newgrp docker    # or open a new SSH session" >&2
    echo "  ./scripts/docker-setup.sh" >&2
    exit 1
  fi
  docker_access=sg
  echo "note: the docker group is not active in this shell, so pulls run through"
  echo "sg docker. open a new SSH session before ./scripts/docker-run.sh."
fi

# The toolkit check reads the package status directly. `dpkg -l | grep -q`
# exits early, which can kill dpkg with SIGPIPE and, under pipefail, report a
# package that is installed as missing. That made the script reinstall the
# toolkit and rewrite its apt source on every run.
toolkit_status="$(dpkg-query -W -f='${Status}' nvidia-container-toolkit 2>/dev/null || true)"
toolkit_installed=0
case "$toolkit_status" in
  *"install ok installed"*) toolkit_installed=1 ;;
esac

# A tarball install has nvidia-ctk but no dpkg entry. Accept it too, so a box
# that is already usable is not sent back through apt.
if [ "$toolkit_installed" -eq 0 ] && command -v nvidia-ctk >/dev/null 2>&1; then
  toolkit_installed=1
fi

if [ "$toolkit_installed" -eq 0 ]; then
  if [ "$root_ok" -eq 0 ]; then
    die "the NVIDIA container toolkit is missing and this shell has no root access. run:"$'\n'"  curl -fsSL -o /tmp/nvidia-ctk.key $gpgkey_url"$'\n'"  sudo gpg --dearmor --yes -o $keyring /tmp/nvidia-ctk.key"$'\n'"  curl -fsSL -o /tmp/nvidia-ctk.list $toolkit_url"$'\n'"  sed 's#deb https://#deb [signed-by=$keyring] https://#g' /tmp/nvidia-ctk.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"$'\n'"  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"$'\n'"  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"$'\n'"then rerun this script"
  fi
  echo "installing the NVIDIA container toolkit"
  if ! command -v gpg >/dev/null 2>&1; then
    die "gpg is missing, so the toolkit signing key cannot be installed. run: sudo apt-get install -y gnupg"
  fi
  if ! curl -fsSL -o /tmp/nvidia-ctk.key "$gpgkey_url"; then
    die "could not download $gpgkey_url. check the network, then run: curl -fsSL -o /tmp/nvidia-ctk.key $gpgkey_url"
  fi
  if ! as_root mkdir -p /usr/share/keyrings ||
    ! as_root gpg --dearmor --yes -o "$keyring" /tmp/nvidia-ctk.key; then
    die "could not install the toolkit signing key. run: sudo gpg --dearmor --yes -o $keyring /tmp/nvidia-ctk.key"
  fi
  if ! curl -fsSL -o /tmp/nvidia-ctk.list "$toolkit_url"; then
    die "could not download $toolkit_url. check the network, then run: curl -fsSL -o /tmp/nvidia-ctk.list $toolkit_url"
  fi
  if ! sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
    /tmp/nvidia-ctk.list > /tmp/nvidia-ctk.signed.list; then
    die "could not add the signing key path to $toolkit_url. run: sed 's#deb https://#deb [signed-by=$keyring] https://#g' /tmp/nvidia-ctk.list"
  fi
  if ! as_root cp /tmp/nvidia-ctk.signed.list /etc/apt/sources.list.d/nvidia-container-toolkit.list; then
    die "could not write the toolkit apt source. run: sudo cp /tmp/nvidia-ctk.signed.list /etc/apt/sources.list.d/nvidia-container-toolkit.list"
  fi
  if ! as_root apt-get update -qq; then
    die "apt-get update failed after the toolkit source was added. run: sudo apt-get update"
  fi
  if ! as_root apt-get install -y -qq nvidia-container-toolkit; then
    die "the toolkit install failed. run: sudo apt-get install -y nvidia-container-toolkit"
  fi
else
  echo "the NVIDIA container toolkit is already installed"
fi

if docker_runtime_has_nvidia; then
  echo "nvidia runtime is configured"
else
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    die "the nvidia runtime is missing from docker and nvidia-ctk is not installed. run:"$'\n'"  sudo apt-get install -y nvidia-container-toolkit"$'\n'"  sudo nvidia-ctk runtime configure --runtime=docker"$'\n'"  sudo systemctl restart docker"
  fi
  if [ "$root_ok" -eq 0 ]; then
    die "the nvidia runtime is not configured and this shell has no root access. run:"$'\n'"  sudo nvidia-ctk runtime configure --runtime=docker"$'\n'"  sudo systemctl restart docker"$'\n'"then rerun this script"
  fi
  echo "configuring the nvidia runtime"
  if ! as_root nvidia-ctk runtime configure --runtime=docker; then
    die "nvidia-ctk could not write the docker runtime config. run: sudo nvidia-ctk runtime configure --runtime=docker"
  fi
  if ! as_root systemctl restart docker >/dev/null 2>&1 &&
      ! as_root service docker restart >/dev/null 2>&1; then
    die "the docker restart after the runtime change failed. run: sudo systemctl restart docker"
  fi
  if ! wait_for_docker; then
    die "the docker daemon did not come back after the restart. run: sudo systemctl status docker"
  fi
  if ! docker_runtime_has_nvidia; then
    die "the nvidia runtime is still absent from docker. run:"$'\n'"  sudo nvidia-ctk runtime configure --runtime=docker"$'\n'"  sudo systemctl restart docker"$'\n'"then rerun this script"
  fi
  echo "configured the nvidia runtime"
fi

# One gate before any registry call. If the group change did not take effect,
# stop here with the fix instead of letting login or pull print a bare
# permission error.
if ! docker_cli info >/dev/null 2>&1; then
  echo "docker-setup.sh: docker commands still cannot reach the daemon." >&2
  echo "run these, then rerun this script:" >&2
  echo "  newgrp docker    # or open a new SSH session" >&2
  echo "  ./scripts/docker-setup.sh" >&2
  exit 1
fi

if [ -n "${GHCR_TOKEN:-}" ]; then
  if ! printf '%s' "$GHCR_TOKEN" | docker_cli login "$registry" \
      -u "${GHCR_USER:-jpbarona}" --password-stdin; then
    die "the registry login to $registry failed. check that GHCR_TOKEN has read:packages, then rerun: GHCR_TOKEN=... ./scripts/docker-setup.sh"
  fi
fi

echo "pulling $image"
if ! docker_cli pull "$image"; then
  echo "docker-setup.sh: the pull of $image failed." >&2
  echo "if it said unauthorized, rerun with: GHCR_TOKEN=... ./scripts/docker-setup.sh" >&2
  if [ "$docker_access" = sg ]; then
    echo "if it said permission denied, run: newgrp docker, then rerun this script" >&2
  fi
  exit 1
fi

echo "pulling $test_image"
if ! docker_cli pull "$test_image"; then
  die "the pull of $test_image failed. run: docker pull $test_image"
fi

echo
echo "READY. start training with:"
if [ "$docker_access" = sg ]; then
  echo "  newgrp docker    # or open a new SSH session, then:"
fi
echo "  ./scripts/docker-run.sh code-score --config configs/lfm-spread.toml --runs-root runs"
