#!/usr/bin/env bash
# Docker CE on Debian 12 + Habana container runtime. Run as root: sudo bash ~/setup/03-docker.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
export DEBIAN_FRONTEND=noninteractive
TARGET_USER="${SUDO_USER:-gaudi2}"

install -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Register the habana runtime (installed by habanalabs-container-runtime in 02)
cat > /etc/docker/daemon.json <<'JSON'
{
  "runtimes": {
    "habana": {
      "path": "/usr/bin/habana-container-runtime",
      "runtimeArgs": []
    }
  }
}
JSON
systemctl enable --now docker
systemctl restart docker
usermod -aG docker "$TARGET_USER"
docker info | grep -A3 -i runtimes
echo
echo "=== 03-docker done === ($TARGET_USER must log out/in, or run 'newgrp docker', before using docker without sudo)"
