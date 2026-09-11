#!/usr/bin/env bash
# Tokens/s benchmarks with vLLM on Gaudi2 inside the 'vllm-gaudi-stable' container.
# Usage: bash experiments/08-bench.sh [model] [tensor-parallel] [cards]
#   bash 08-bench.sh Qwen/Qwen2.5-7B-Instruct 1 0          # one card
#   bash 08-bench.sh Qwen/Qwen2.5-32B-Instruct 4 4,5,6,7   # four cards
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"; TP="${2:-1}"; CARDS="${3:-0}"; C=vllm-gaudi-stable
run() { docker exec -e HABANA_VISIBLE_DEVICES="$CARDS" -e PT_HPU_LAZY_MODE=1 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true $C bash -c "cd /models && $*"; }
echo "=== batch throughput (128 prompts, 512 in / 256 out) ==="
run vllm bench throughput --model "$MODEL" --dtype bfloat16 --tensor-parallel-size $TP \
    --input-len 512 --output-len 256 --num-prompts 128 --max-model-len 2048 --max-num-seqs 128 2>&1 | grep -E '^Throughput|Error|OOM'
echo "=== single user latency (batch 1, 128 in / 256 out) ==="
run vllm bench latency --model "$MODEL" --dtype bfloat16 --tensor-parallel-size $TP --batch-size 1 \
    --input-len 128 --output-len 256 --num-iters-warmup 3 --num-iters 10 --max-model-len 2048 \
    --max-num-seqs 8 --gpu-memory-utilization 0.5 2>&1 | grep -E 'Avg latency|Error|OOM' | \
    awk '/Avg latency/{printf "Avg latency: %.2f s  =>  ~%.0f tokens/s for one user\n",$3,256/$3} !/Avg latency/{print}'
