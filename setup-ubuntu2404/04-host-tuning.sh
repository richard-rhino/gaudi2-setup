#!/usr/bin/env bash
# Host tuning for Gaudi/HCCL (same as the Debian setup). sudo bash setup-ubuntu2404/04-host-tuning.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
# Hugepages: Intel installer formula (110 MB x cores x 2). Synapse uses them for host buffers.
N=$(( (110*1024*$(nproc)*2) / $(grep '^Hugepagesize:' /proc/meminfo | awk '{print $2}') + 1 ))
sysctl -w vm.nr_hugepages=$N
echo "vm.nr_hugepages=$N" > /etc/sysctl.d/90-habanalabs-hugepages.conf
# Unlimited locked memory: HCCL init fails for a second process without it. Applies to new logins.
cat > /etc/security/limits.d/90-habanalabs.conf <<'EOF'
*    soft memlock unlimited
*    hard memlock unlimited
*    soft nofile  1048576
*    hard nofile  1048576
EOF
# No core dumps: a crashed vLLM worker writes ~10 GB cores.
echo 'kernel.core_pattern=|/bin/false' > /etc/sysctl.d/90-no-core.conf && sysctl -p /etc/sysctl.d/90-no-core.conf
grep -i HugePages_Total /proc/meminfo
