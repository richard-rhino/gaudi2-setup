#!/usr/bin/env bash
# Intel Gaudi 1.24.1 driver + tools on Debian 12, using Intel's Ubuntu 22.04 (jammy) apt repo.
# That repo contains ONLY habanalabs-* packages, so it cannot clobber Debian packages.
# Run as root:  sudo bash ~/setup/02-gaudi.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
export DEBIAN_FRONTEND=noninteractive
VER=1.24.1-482

[[ -d /lib/modules/$(uname -r)/build ]] || { echo "No kernel headers for running kernel $(uname -r). Reboot into the newest kernel or install linux-headers-$(uname -r)."; exit 1; }

install -d /etc/apt/keyrings
curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public | gpg --dearmor -o /etc/apt/keyrings/habanalabs.gpg
echo "deb [signed-by=/etc/apt/keyrings/habanalabs.gpg] https://vault.habana.ai/artifactory/debian jammy main" \
  > /etc/apt/sources.list.d/habanalabs.list
apt-get update

# Debian 12 patch: Intel's header-probe script mis-handles Debian's absolute-path headers Makefile.
# The DKMS build fails on first install; we patch the unpacked source, then re-run the postinst.
apply_patch() {
  local f=/usr/src/habanalabs-${VER}/compat/scripts/generate_flags.sh
  [[ -f $f ]] || return 0
  grep -q 'ABSOLUTE path' "$f" && return 0
  cp "$f" "$f.orig"
  cp "$(dirname "$0")/generate_flags.sh.patched" "$f"
  echo "patched $f"
}

# Pin the exact release so a later `apt upgrade` doesn't silently move the driver.
apt-get install -y \
  habanalabs-dkms=$VER \
  habanalabs-firmware=$VER \
  habanalabs-firmware-tools=$VER \
  habanalabs-rdma-core=$VER \
  habanalabs-thunk=$VER \
  habanalabs-graph=$VER \
  habanalabs-container-runtime=$VER || true   # dkms postinst is expected to fail on Debian the first time
apply_patch
if ! dkms status 2>/dev/null | grep -q "habanalabs/${VER}.*installed"; then
  dkms remove habanalabs/${VER} --all >/dev/null 2>&1 || true
  dpkg --configure habanalabs-dkms
  apt-get install -f -y
fi
apt-mark hold habanalabs-dkms habanalabs-firmware habanalabs-firmware-tools habanalabs-rdma-core habanalabs-thunk habanalabs-graph habanalabs-container-runtime


# Hugepages, same formula as Intel's habanalabs-installer.sh (110 MB x cores x 2). Synapse uses them for host buffers.
N=$(( (110*1024*$(nproc)*2) / $(grep '^Hugepagesize:' /proc/meminfo | awk '{print $2}') + 1 ))
sysctl -w vm.nr_hugepages=$N
echo "vm.nr_hugepages=$N" > /etc/sysctl.d/90-habanalabs-hugepages.conf

# Intel requires IOMMU passthrough. Without iommu=pt, HPU compute silently returns wrong results on this box.
if ! grep -q 'iommu=pt' /etc/default/grub; then
  cp /etc/default/grub /etc/default/grub.bak-$(date +%s)
  sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 intel_iommu=on iommu=pt"/' /etc/default/grub
  update-grub
  echo ">>> GRUB updated with intel_iommu=on iommu=pt: REBOOT before using the cards."
fi

echo "=== dkms status ==="
dkms status
grep -qx habanalabs /etc/modules-load.d/habanalabs.conf || sed -i "1a habanalabs" /etc/modules-load.d/habanalabs.conf
echo "=== loading driver ==="
modprobe habanalabs_compat && modprobe habanalabs && modprobe habanalabs_cn && modprobe habanalabs_en && modprobe habanalabs_ib || { echo "modprobe failed; see: dmesg | grep -i habana  and  /var/lib/dkms/habanalabs/*/build/make.log"; exit 1; }
sleep 5
ls -l /dev/accel/ || true
echo "=== hl-smi ==="
hl-smi || echo "hl-smi failed; check dmesg | grep -i habana"
echo
echo "=== 02-gaudi done ==="
echo "If hl-smi shows all 8 cards, continue with 03-docker.sh"
