#!/usr/bin/env bash
# Stop a native (non-Docker) vLLM server on Ubuntu and wait until the Gaudi cards are released.
# SIGINT is ignored while vLLM is warming up, so: TERM, wait, then KILL.
pkill -TERM -f "[b]in/vllm serve|[V]LLM::" 2>/dev/null; sleep 15
pkill -9 -f "[b]in/vllm serve|[b]in/vllm bench|[V]LLM::|[r]esource_tracker" 2>/dev/null
for i in $(seq 1 30); do
  busy=$(hl-smi -Q memory.used -f csv,noheader | awk '{s+=($1>2000)} END{print s+0}')
  [[ "$busy" == "0" ]] && { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null; echo "cards released"; exit 0; }
  sleep 2
done
echo "WARNING: cards still busy"; hl-smi -Q index,memory.used -f csv; exit 1
