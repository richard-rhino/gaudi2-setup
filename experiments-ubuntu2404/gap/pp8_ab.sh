#!/usr/bin/env bash
# Default PP=8 TP=1 no MTP (override with TP=.. PP=..), one launch per config: LABEL ENV... ; single-user speed (repo script), greedy eq3 (alone/batched),
# host timing lines and PLE host-check lines. Usage: pp8_ab.sh LABEL [VAR=VAL ...]
S=$(cd "$(dirname "$0")" && pwd)
cd ~/gaudi2-setup
OUT=$S/pp8_ab.out
lab=$1; shift
[ -z "$(sudo fuser /dev/accel/* 2>/dev/null)" ] || { echo "cards busy" | tee -a $OUT; exit 9; }
trap 'bash setup-ubuntu2404/stop-vllm.sh' EXIT
L=logs/pp8ab-$lab-$(date +%m%d-%H%M).log
# $S/hosttime holds hosttime_sitecustomize.py saved as sitecustomize.py (host timers per step)
env PYTHONPATH=$S/hosttime MTP=0 TP=1 PP=8 "$@" bash experiments-ubuntu2404/qwen38fn-serve.sh > $L 2>&1 &
pid=$!
for i in $(seq 1 300); do curl -sf -m 3 localhost:8000/v1/models >/dev/null && break; kill -0 $pid 2>/dev/null || { echo "$lab server exited" | tee -a $OUT; tail -5 $L; exit 1; }; sleep 5; done
echo "#### $lab $* $(date +%H:%M:%S) plugin $(git -C ~/src/vllm-gaudi rev-parse --short HEAD)+$(git -C ~/src/vllm-gaudi diff | md5sum | cut -c1-8)" | tee -a $OUT
curl -s -m 1500 localhost:8000/v1/completions -H 'Content-Type: application/json' -o /dev/null -w "warmup %{time_total} s\n" \
  -d '{"model":"qwen38fn","prompt":"Write a long story about a lighthouse keeper.\n","max_tokens":256,"temperature":0,"ignore_eos":true}'
for r in 1 2; do timeout 2400 python3 experiments-ubuntu2404/single_user_speed.py http://localhost:8000 qwen38fn 2>&1 | tail -3 | sed "s/^/speed $lab /" | tee -a $OUT; done
timeout 2400 python3 $S/pp8_eq3.py run $lab 2>&1 | tee -a $OUT
grep -E 'HOSTTIME rank (0|7) ' $L | sed 's/^.*HOSTTIME/HOSTTIME/' | grep -vE 'execute_model:|gap_between_sample_tokens: median 2[0-9]' \
  | awk '{k=$3" "$4; last[k]=$0} END{for(k in last) print last[k]}' | sort | sed "s/^/$lab /" | tee -a $OUT
grep -E 'PLE host tokens check|PLE prep host' $L | tail -2 | sed 's/^.*INFO/INFO/;s/^/'"$lab"' /' | tee -a $OUT
echo "errors $(grep -cE 'Traceback|Error' $L) log $L" | tee -a $OUT
