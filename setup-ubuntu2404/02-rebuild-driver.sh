#!/usr/bin/env bash
# Rebuild Intel Gaudi DKMS driver for the running kernel.  sudo bash ~/setup/02-rebuild-driver.sh && sudo reboot
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
K=$(uname -r)
# meta-package so future kernel upgrades pull headers and DKMS rebuilds automatically
DEBIAN_FRONTEND=noninteractive apt-get install -y linux-headers-generic "linux-headers-$K"
dkms install habanalabs/1.24.2-481 -k "$K" --force
dkms status | grep habana
ls /lib/modules/"$K"/updates/dkms/
depmod -a "$K"
# system libs needed by the PyTorch-for-Gaudi install (libnuma, openblas, lapack, perftools, ...)
cd /home/ubun/setup/installer && ./habanalabs-installer.sh install --type dependencies -y
echo "DONE. Now: sudo reboot"
