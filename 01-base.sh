#!/usr/bin/env bash
# Base packages for a fresh Debian 12 box + everything the Habana packages depend on.
# Run as root:  sudo bash ~/setup/01-base.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
export DEBIAN_FRONTEND=noninteractive
TARGET_USER="${SUDO_USER:-gaudi2}"

apt-get update
apt-get -y full-upgrade

apt-get install -y \
  build-essential gcc g++ make cmake ninja-build pkg-config git curl wget gpg ca-certificates rsync parted \
  dkms linux-headers-amd64 libelf-dev \
  python3 python3-pip python3-venv python3-dev \
  zsh tmux htop btop ncdu fzf ripgrep fd-find bat tree jq unzip less vim \
  pciutils ethtool file libibverbs-dev \
  libboost-dev libfdt-dev libudev-dev libnl-3-dev libnl-route-3-dev valgrind libsystemd-dev \
  libcurl4-openssl-dev libsox-dev libboost-filesystem-dev libboost-system-dev libssl-dev \
  libomp5 libomp-dev libdrm-dev libopenblas-dev

# zsh as the login shell (chsh would ask for a password when run as the user)
chsh -s /usr/bin/zsh "$TARGET_USER"

# Debian names these binaries oddly; give them their usual names for the user
ln -sf /usr/bin/batcat /usr/local/bin/bat
ln -sf /usr/bin/fdfind /usr/local/bin/fd

NEWEST_KERNEL=$(ls -1 /lib/modules | sort -V | tail -1)
echo
echo "=== 01-base done ==="
if [[ "$NEWEST_KERNEL" != "$(uname -r)" ]]; then
  echo "A newer kernel ($NEWEST_KERNEL) was installed; you are running $(uname -r)."
  echo ">>> REBOOT before running 02-gaudi.sh so the driver builds for the kernel you actually run."
else
  echo "Kernel unchanged ($(uname -r)); continue with 02-gaudi.sh"
fi
