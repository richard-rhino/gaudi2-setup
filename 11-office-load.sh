#!/usr/bin/env bash
# "Office" serving test: one model across all 8 Gaudi2 cards behind the OpenAI-compatible server,
# then realistic concurrent load with vllm's serving benchmark. Results -> ~/setup/office-load-results.md
# Usage: bash ~/setup/11-office-load.sh [model] [max-model-len]
MODEL="${1:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"; MAXLEN="${2:-32768}"; TP="${TP:-8}"; EXTRA="${EXTRA:-}"; CARDS="${CARDS:-all}"; C=vllm-gaudi-stable; PORT=8000
NAME=$(echo "$MODEL" | tr "/" "_")-tp${TP}$(echo "$EXTRA" | tr -dc "a-z0-9" | cut -c1-12); SLOG=~/setup/serve-$NAME.log; OUT=~/setup/office-load-results-$NAME.md

echo "=== starting server: $MODEL TP=$TP cards=$CARDS extra=[$EXTRA] max_model_len=$MAXLEN ==="
bash ~/setup/stop-vllm.sh || exit 1
docker exec -d -e HABANA_VISIBLE_DEVICES=$CARDS -e PT_HPU_LAZY_MODE=1 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  -e VLLM_GRAPH_RESERVED_MEM=0.1 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 -e VLLM_RPC_TIMEOUT=100000 \
  $C bash -c "cd /models && vllm serve '$MODEL' --served-model-name office --tensor-parallel-size $TP $EXTRA \
    --max-model-len $MAXLEN --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 --port $PORT > /root/serve.log 2>&1"
t0=$(date +%s)
for i in $(seq 1 720); do   # up to 2 h for weight load + warmup
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
  if docker exec $C grep -qE 'Traceback|Engine core initialization failed' /root/serve.log 2>/dev/null; then
    echo "SERVER FAILED:"; docker exec $C grep -E 'RuntimeError|ValueError|Error:' /root/serve.log | tail -5; exit 1; fi
  sleep 10
done
docker exec $C cp /root/serve.log /root/serve-$NAME.log; docker cp $C:/root/serve-$NAME.log "$SLOG" 2>/dev/null
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "server not ready after 2 h"; exit 1; }
echo "server ready after $(( ($(date +%s)-t0)/60 )) min (load + warmup)"
grep -E 'Warmup finished|KV cache size|Loading weights took' "$SLOG" | tail -3

echo "=== sanity: one real chat request ==="
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"office","messages":[{"role":"user","content":"In one sentence, what is Intel Gaudi?"}],"max_tokens":60}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'

[[ -f $OUT ]] || printf "# Office load test: %s, TP=%s cards=%s extra=[%s], max_model_len %s\n\n| Scenario | Requests | Concurrency | Input/Output tokens | Req/s | Output tok/s | Total tok/s | TTFT p50/p99 (s) | TPOT p50/p99 (ms) | E2E p50/p99 (s) |\n|---|---|---|---|---|---|---|---|---|---|\n" "$MODEL" "$TP" "$CARDS" "$EXTRA" "$MAXLEN" > $OUT
bench() { # name num_prompts concurrency(or inf) in out
  local name=$1 n=$2 conc=$3 in=$4 out=$5 extra=""
  [[ "$conc" == "inf" ]] && extra="--request-rate inf" || extra="--max-concurrency $conc --request-rate inf"
  echo "=== $name: $n requests, concurrency $conc, ${in} in / ${out} out ==="
  docker exec $C bash -c "cd /models && vllm bench serve --backend vllm --host 127.0.0.1 --port $PORT --model '$MODEL' --served-model-name office \
     --dataset-name random --random-input-len $in --random-output-len $out --random-range-ratio 0.2 --num-prompts $n $extra \
     --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --seed 42 --disable-tqdm" > ~/setup/office-$name.log 2>&1
  local L=~/setup/office-$name.log
  g() { grep -E "$1" "$L" | tail -1 | awk '{print $NF}'; }
  local rps=$(g 'Request throughput') otps=$(g 'Output token throughput') ttps=$(g 'Total token throughput')
  local t50=$(g 'Median TTFT') t99=$(g 'P99 TTFT') p50=$(g 'Median TPOT') p99=$(g 'P99 TPOT') e50=$(g 'Median E2EL') e99=$(g 'P99 E2EL')
  local ttft=$(python3 -c "print(f'{${t50:-0}/1000:.2f} / {${t99:-0}/1000:.2f}')" 2>/dev/null); local e2e=$(python3 -c "print(f'{${e50:-0}/1000:.1f} / {${e99:-0}/1000:.1f}')" 2>/dev/null)
  grep -qE 'Output token throughput' "$L" || { echo "FAILED: $(grep -iE 'error' "$L" | tail -1)"; rps=fail; }
  printf '| %s | %s | %s | %s / %s | %s | %s | %s | %s | %s / %s | %s |\n' "$name" "$n" "$conc" "$in" "$out" "$rps" "$otps" "$ttps" "$ttft" "$p50" "$p99" "$e2e" | tee -a $OUT
}
bench warmup-run      16  8    512   128   >/dev/null
bench burst-64-at-once 64  inf  1024  256
bench sustained-32     128 32   2048  512
bench sustained-64     192 64   2048  512
bench long-ctx-8       16  8    12288 512
bench long-ctx-16      32  16   8192  512
echo; echo "=== done: results in $OUT (server left running on port $PORT; stop with: docker exec $C pkill -f 'vllm serve') ==="
