#!/usr/bin/env bash
# Sample every PP worker with py-spy during a single-user decode run.
# usage: pyspy_pp_workers.sh OUTDIR [seconds]
set -u
OUT=$1; DUR=${2:-20}; mkdir -p "$OUT"
cd "$(dirname "$0")/.."
python3 experiments-ubuntu2404/single_user_speed.py http://localhost:8000 qwen38fn > "$OUT/speed.txt" 2>&1 &
SP=$!
sleep 25   # skip the first run's TTFT
PIDS=$(ps -eo pid,args | awk '$2 ~ /^VLLM::Worker_PP[0-9]+$/ {print $1}')
for p in $PIDS; do
  name=$(ps -o args= -p $p | grep -oE 'PP[0-9]+' | head -1)
  sudo ~/vllm-venv/bin/py-spy record -p $p -d $DUR -r 100 -f raw -o "$OUT/${name:-pid$p}.raw" --nonblocking >/dev/null 2>&1 &
done
wait $SP
wait
ls "$OUT"; tail -3 "$OUT/speed.txt"
