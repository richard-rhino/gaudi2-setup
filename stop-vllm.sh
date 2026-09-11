#!/usr/bin/env bash
# Stop every vLLM server/worker in all containers and wait until the Gaudi cards are actually released.
for c in $(docker ps --format '{{.Names}}' | grep -E '^vllm-gaudi'); do
  docker exec $c bash -c 'pkill -9 -f "[v]llm serve|[v]llm bench|[V]LLM::|EngineCore|Worker_TP" 2>/dev/null; true'
done
for i in $(seq 1 30); do
  busy=$(hl-smi -Q memory.used -f csv,noheader 2>/dev/null | awk '{s+=($1>2000)} END{print s+0}')
  [[ "$busy" == "0" ]] && { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null; echo "cards released, page cache dropped (driver needs contiguous low DMA memory for new contexts)"; exit 0; }
  sleep 2
done
echo "WARNING: cards still busy:"; hl-smi -Q index,memory.used -f csv | head -9
echo "host processes holding /dev/accel:"; for p in /proc/[0-9]*; do sudo ls -l $p/fd 2>/dev/null | grep -q '/dev/accel' && echo "  pid ${p#/proc/} $(sudo tr '\0' ' ' < $p/cmdline | cut -c1-60)"; done
exit 1
