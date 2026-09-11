#!/usr/bin/env bash
# Requirement test: >=10 concurrent users, each >=30 tok/s, 128k context, mixed chat + agentic traffic.
# Usage: TP=8 EXTRA="--enable-expert-parallel" bash experiments/12-office-10users.sh <model> [max_model_len=131072]
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?model}"; MAXLEN="${2:-131072}"; TP="${TP:-8}"; EXTRA="${EXTRA:-}"; CARDS="${CARDS:-all}"; C="${C:-vllm-gaudi-stable}"; PORT=8000
NAME=$(echo "$MODEL" | tr '/' '_')-tp${TP}-$(echo "$EXTRA" | tr -dc 'a-z0-9' | cut -c1-10)-ctx$MAXLEN-mem${GPU_MEM:-0.85}; OUT=$ROOT/results/office10-$NAME.md
echo "=== server: $MODEL TP=$TP extra=[$EXTRA] max_model_len=$MAXLEN ==="
bash "$ROOT"/setup/stop-vllm.sh || exit 1
# long-context settings from the plugin docs; exp bucketing prepares long buckets automatically
docker exec -d -e HABANA_VISIBLE_DEVICES=$CARDS -e PT_HPU_LAZY_MODE=${LAZY:-1} -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  -e VLLM_GRAPH_RESERVED_MEM=${GRAPH_MEM:-0.1} -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 -e VLLM_RPC_TIMEOUT=100000 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  $C bash -c "cd /models && vllm serve '$MODEL' --served-model-name office --tensor-parallel-size $TP $EXTRA \
    --max-model-len $MAXLEN --max-num-seqs ${MAX_SEQS:-32} --max-num-batched-tokens 16384 --gpu-memory-utilization ${GPU_MEM:-0.85} \
    --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER:-hermes} --host 0.0.0.0 --port $PORT > /root/serve.log 2>&1"
t0=$(date +%s)
for i in $(seq 1 1080); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
  docker exec $C grep -aqE 'Traceback|Engine core initialization failed|vllm serve: error|usage: vllm serve' /root/serve.log 2>/dev/null && { echo "SERVER FAILED:"; docker exec $C grep -aE 'RuntimeError|ValueError|Error:|Error\b|vllm serve: error' /root/serve.log | grep -vE 'vllm._C|commit hash' | tail -4 | cut -c1-200; docker cp $C:/root/serve.log "$ROOT"/logs/serve/serve-FAILED-$NAME.log 2>/dev/null; exit 1; }
  sleep 10; done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "not ready after 3 h"; exit 1; }
READY=$(( ($(date +%s)-t0)/60 )); echo "ready after $READY min"; docker exec $C grep -aE 'KV cache size|Warmup finished' /root/serve.log | tail -2 | cut -c1-140
docker exec $C cp /root/serve.log /root/serve-office10-$NAME.log
printf '# %s  TP=%s extra=[%s]  max_model_len=%s  gpu_mem=%s graph_mem=%s  startup %s min\n\n| Scenario | Users | Prompt/answer tok | Per-user tok/s (p50) | Per-user tok/s (p99 user) | TTFT p50 / p99 (s) | Aggregate out tok/s | Total tok/s |\n|---|---|---|---|---|---|---|---|\n' "$MODEL" "$TP" "$EXTRA" "$MAXLEN" "${GPU_MEM:-0.85}" "${GRAPH_MEM:-0.1}" "$READY" > $OUT
bench() { local name=$1 n=$2 conc=$3 in=$4 out=$5
  echo "=== $name: $n req, $conc users, ${in} in / ${out} out ==="
  docker exec $C bash -c "cd /models && vllm bench serve --backend vllm --host 127.0.0.1 --port $PORT --model '$MODEL' --served-model-name office \
     --dataset-name random --random-input-len $in --random-output-len $out --random-range-ratio 0.2 --num-prompts $n --max-concurrency $conc --request-rate inf \
     --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --seed 7 --disable-tqdm" > "$ROOT"/logs/office/office10-$name-$NAME.log 2>&1
  local L="$ROOT"/logs/office/office10-$name-$NAME.log; g() { grep -E "$1" "$L" | tail -1 | awk '{print $NF}'; }
  local p50=$(g 'Median TPOT') p99=$(g 'P99 TPOT') t50=$(g 'Median TTFT') t99=$(g 'P99 TTFT') o=$(g 'Output token throughput') tt=$(g 'Total token throughput')
  local u50=$(python3 -c "print(round(1000/${p50:-1e9},1))") u99=$(python3 -c "print(round(1000/${p99:-1e9},1))") ttft=$(python3 -c "print(f'{${t50:-0}/1000:.1f} / {${t99:-0}/1000:.1f}')")
  [[ -z "$o" ]] && { o="FAIL: $(grep -iE 'error' "$L" | tail -1 | cut -c1-80)"; }
  printf '| %s | %s | %s / %s | %s | %s | %s | %s | %s |\n' "$name" "$conc" "$in" "$out" "$u50" "$u99" "$ttft" "$o" "$tt" | tee -a $OUT; }
bench warm            8   4   1024   128 >/dev/null
bench chat-10         40  10  1024   512
bench chat-16         48  16  1024   512
bench agentic-10-32k  20  10  32768  1024
bench mixed-16        48  16  8192   768
bench longctx-4-100k  4   4   100000 512
echo "=== done -> $OUT (server still up on :$PORT) ==="
