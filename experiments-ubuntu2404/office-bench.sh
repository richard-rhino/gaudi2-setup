#!/usr/bin/env bash
# Section-8 scenarios against an already-running native server (e.g. qwen38fn-serve.sh), no Docker.
# Each scenario first runs a small warm-up pass (the server may run with VLLM_SKIP_WARMUP, so new shapes compile on
# first use), then the measured pass. Usage: bash office-bench.sh <hf-model> <served-name> <tag>
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?hf model}"; SERVED="${2:?served name}"; TAG="${3:?tag}"; PORT=${PORT:-8000}
source ~/vllm-venv/bin/activate; export HF_HOME=/models/huggingface
OUT=$ROOT/results-ubuntu2404/office-$TAG.md; mkdir -p "$ROOT"/logs/office
printf '# %s (%s) %s\n\n| Scenario | Users | Prompt/answer tok | Per-user tok/s (p50) | Per-user tok/s (p99 user) | TTFT p50 / p99 (s) | Aggregate out tok/s | Total tok/s |\n|---|---|---|---|---|---|---|---|\n' "$MODEL" "$TAG" "$(date -Is)" > $OUT
run() { vllm bench serve --backend vllm --host 127.0.0.1 --port $PORT --model "$MODEL" --served-model-name "$SERVED" \
     --dataset-name random --random-input-len $3 --random-output-len $4 --random-range-ratio 0.2 --num-prompts $1 --max-concurrency $2 --request-rate inf \
     --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --seed ${5:-7} --disable-tqdm; }
bench() { local name=$1 n=$2 conc=$3 in=$4 out=$5 L="$ROOT"/logs/office/$TAG-$1.log
  echo "=== $name: $n req, $conc users, ${in} in / ${out} out  $(date +%T)"
  run $conc $conc $in 32 11 > "$L.warm" 2>&1
  run $n $conc $in $out > "$L" 2>&1
  g() { grep -E "$1" "$L" | tail -1 | awk '{print $NF}'; }
  local p50=$(g 'Median TPOT') p99=$(g 'P99 TPOT') t50=$(g 'Median TTFT') t99=$(g 'P99 TTFT') o=$(g 'Output token throughput') tt=$(g 'Total token throughput')
  local u50=$(python3 -c "print(round(1000/${p50:-1e9},1))") u99=$(python3 -c "print(round(1000/${p99:-1e9},1))") ttft=$(python3 -c "print(f'{${t50:-0}/1000:.1f} / {${t99:-0}/1000:.1f}')")
  [[ -z "$o" ]] && o="FAIL: $(grep -iE 'error' "$L" | tail -1 | cut -c1-80)"
  printf '| %s | %s | %s / %s | %s | %s | %s | %s | %s |\n' "$name" "$conc" "$in" "$out" "$u50" "$u99" "$ttft" "$o" "$tt" | tee -a $OUT; }
for s in ${SCENARIOS:-single chat-10 chat-16 agentic-10-24k mixed-16}; do case $s in
  single)          bench single          5   1  128   256 ;;
  chat-10)         bench chat-10         40  10 1024  512 ;;
  chat-16)         bench chat-16         48  16 1024  512 ;;
  agentic-10-24k)  bench agentic-10-24k  20  10 24576 1024 ;;
  mixed-16)        bench mixed-16        48  16 8192  768 ;;
esac; done
echo "=== done -> $OUT"
