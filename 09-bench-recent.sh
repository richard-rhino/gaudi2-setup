#!/usr/bin/env bash
# Benchmark a list of recent models on Gaudi2 and append results to ~/setup/bench-results.md
# Usage: bash ~/setup/09-bench-recent.sh "MODEL TP CARDS" ["MODEL TP CARDS" ...]
#   bash 09-bench-recent.sh "Qwen/Qwen3.8-27B 1 0" "Qwen/Qwen3.8-27B 4 0,1,2,3" "Qwen/Qwen3.6-35B-A3B 1 4"
C=vllm-gaudi-stable; OUT=~/setup/bench-results.md
[[ -f $OUT ]] || printf '| Model | Cards (TP) | Batched output tok/s (128 prompts, 512in/256out) | Total tok/s | Single-user tok/s | Notes |\n|---|---|---|---|---|---|\n' > $OUT
for spec in "$@"; do
  read -r MODEL TP CARDS <<<"$spec"
  LOG=~/setup/bench-$(echo "$MODEL" | tr '/' '_')-tp$TP.log
  echo "=== $MODEL  TP=$TP cards=$CARDS  (log: $LOG) ==="
  run() { docker exec -e HABANA_VISIBLE_DEVICES="$CARDS" -e PT_HPU_LAZY_MODE=${LAZY:-1} -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true $C bash -c "cd /models && $*"; }
  run vllm bench throughput --model "$MODEL" --dtype bfloat16 --tensor-parallel-size $TP --trust-remote-code \
      --input-len 512 --output-len 256 --num-prompts 128 --max-model-len 2048 --max-num-seqs 128 \
      --gpu-memory-utilization ${GPU_MEM:-0.8} > "$LOG" 2>&1
  T=$(grep -E '^Throughput' "$LOG" | tail -1)
  OUTTOK=$(echo "$T" | grep -oE '[0-9.]+ output tokens/s' | awk '{print $1}')
  TOTTOK=$(echo "$T" | grep -oE '[0-9.]+ total tokens/s' | awk '{print $1}')
  ERR=$(grep -vE 'lazy mode' "$LOG" | grep -oE 'RuntimeError: .{0,80}|ValueError: .{0,80}|OOM.{0,40}|Error: .{0,80}' | tail -1)
  run vllm bench latency --model "$MODEL" --dtype bfloat16 --tensor-parallel-size $TP --trust-remote-code --batch-size 1 \
      --input-len 128 --output-len 256 --num-iters-warmup 3 --num-iters 10 --max-model-len 2048 \
      --max-num-seqs 8 --gpu-memory-utilization 0.5 >> "$LOG" 2>&1
  LAT=$(grep -E '^Avg latency' "$LOG" | tail -1 | awk '{print $3}')
  SU=$([[ -n "$LAT" ]] && python3 -c "print(round(256/$LAT))" || echo "-")
  printf '| %s | %s (%s) | %s | %s | %s | %s |\n' "$MODEL" "$CARDS" "$TP" "${OUTTOK:-fail}" "${TOTTOK:-fail}" "$SU" "${ERR:-ok}" | tee -a $OUT
done
