#!/usr/bin/env bash
# PP=8, no MTP, one user: device profile of ~8 warm decode steps (torch profiler, HPU events)
S=$(cd "$(dirname "$0")" && pwd)
cd ~/gaudi2-setup
D=$S/devprof/pp8mtp-$(date +%m%d-%H%M); mkdir -p $D
[ -z "$(sudo fuser /dev/accel/* 2>/dev/null)" ] || { echo "cards busy"; exit 9; }
trap 'bash setup-ubuntu2404/stop-vllm.sh' EXIT
: > logs/.serve-current.out
PROFILE=$D MTP=3 TP=1 PP=8 bash experiments-ubuntu2404/qwen38fn-serve.sh > logs/.serve-current.out 2>&1 &
pid=$!
for i in $(seq 1 300); do curl -sf -m 3 localhost:8000/v1/models >/dev/null && break; kill -0 $pid 2>/dev/null || { echo "server exited"; exit 1; }; sleep 5; done
q() { curl -s -m 1200 localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen38fn\",\"prompt\":\"Write a long story about a lighthouse keeper.\\n\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
  -o /dev/null -w "request $1 tokens: %{time_total} s\n"; }
q 256; q 256; q 9
curl -s -X POST localhost:8000/start_profile -w 'start %{http_code}\n'
q 9
curl -s -m 1200 -X POST localhost:8000/stop_profile -w 'stop %{http_code}\n'
for i in $(seq 1 120); do [ "$(ls $D | grep -c '^trace-')" -ge 8 ] && break; sleep 5; done
mkdir -p $D/session1; mv $D/trace-* $D/session1/
q 9
curl -s -X POST localhost:8000/start_profile -w 'start2 %{http_code}\n'
q 13
curl -s -m 1200 -X POST localhost:8000/stop_profile -w 'stop2 %{http_code}\n'
for i in $(seq 1 120); do [ "$(ls $D | grep -c '^trace-')" -ge 8 ] && break; sleep 5; done
ls -la $D | head -20; echo "dir $D"
