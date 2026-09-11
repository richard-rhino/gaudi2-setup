#!/usr/bin/env bash
# Post-reboot check: kernel params, driver, then HPU-vs-CPU numeric test inside the container.
set -uo pipefail
echo "== cmdline =="; cat /proc/cmdline
echo "== iommu mode =="; sudo dmesg | grep -iE 'iommu: Default domain|DMAR: IOMMU enabled' | head -3
echo "== driver =="; lsmod | grep -c habanalabs; hl-smi -Q index,name,memory.total -f csv | head -3
echo "== containers =="; docker ps --format '{{.Names}} {{.Status}}'
echo "== HPU numeric test (rel err should be ~1e-6 everywhere) =="
docker cp ~/setup/hpu-size.py vllm-gaudi-stable:/tmp/hpu-size.py
docker exec -e HABANA_VISIBLE_DEVICES=0 vllm-gaudi-stable bash -c 'cd /tmp && timeout 900 python3 hpu-size.py 2>&1 | grep -E "gather|rel err|run [0-9]|stage"'
