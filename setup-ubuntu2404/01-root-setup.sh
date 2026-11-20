#!/usr/bin/env bash
# Root-level setup for Gaudi2 + vLLM. Run with:  sudo bash ~/setup/01-root-setup.sh
# Idempotent: safe to re-run. Reboot when it finishes.
set -euo pipefail

GAUDI_VER=1.24.2
TARGET_USER=ubun
NVME_UUID=edfe62f1-e0d9-4f03-b0f3-2408eb4cc045
MOUNTPOINT=/models

[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }

echo "==> [1/6] apt packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  zsh zsh-autosuggestions zsh-syntax-highlighting \
  build-essential python3-dev python3-venv python3-pip \
  git wget curl ca-certificates pciutils ethtool \
  linux-headers-"$(uname -r)" dkms

echo "==> [2/6] mount NVMe (label nvme-models) at $MOUNTPOINT"
mkdir -p "$MOUNTPOINT"
if ! grep -q "$NVME_UUID" /etc/fstab; then
  echo "UUID=$NVME_UUID $MOUNTPOINT ext4 defaults,noatime,nofail 0 2" >> /etc/fstab
fi
systemctl daemon-reload
mountpoint -q "$MOUNTPOINT" || mount "$MOUNTPOINT"
df -h "$MOUNTPOINT"

echo "==> [3/6] device access: add $TARGET_USER to render group"
usermod -aG render "$TARGET_USER"

echo "==> [4/6] IOMMU passthrough (required by Intel for Ubuntu 24.04 / kernel 6.8)"
if ! grep -q 'iommu=pt' /etc/default/grub; then
  sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 iommu=pt intel_iommu=on"/' /etc/default/grub
  update-grub
fi
grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub

echo "==> [5/6] Intel Gaudi base software $GAUDI_VER (driver/firmware/runtime)"
WORK=/root/gaudi-install-$GAUDI_VER
mkdir -p "$WORK" && cd "$WORK"
[[ -f habanalabs-installer.sh ]] || wget -nv "https://vault.habana.ai/artifactory/gaudi-installer/$GAUDI_VER/habanalabs-installer.sh"
chmod +x habanalabs-installer.sh
./habanalabs-installer.sh install --type base -y

echo "==> [6/6] default shell -> zsh for $TARGET_USER"
chsh -s /usr/bin/zsh "$TARGET_USER"

echo
echo "DONE. Now reboot:  sudo reboot"
echo "After reboot, check:  hl-smi"
