#!/usr/bin/env bash
# Native (no Docker) port of experiments/08-bench.sh for the Ubuntu 24.04 host.
# Usage: bash bench.sh <model> <tp> <cards> [lazy=1|0]
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"; TP="${2:-1}"; CARDS="${3:-0}"; LAZY="${4:-1}"
source ~/vllm-venv/bin/activate
export HF_HOME=/models/huggingface HABANA_VISIBLE_DEVICES="$CARDS" PT_HPU_LAZY_MODE=$LAZY PT_HPU_ENABLE_LAZY_COLLECTIVES=true
echo "=== $MODEL tp=$TP cards=$CARDS lazy=$LAZY $(date -Is)"
echo "=== batch throughput (128 prompts, 512 in / 256 out) ==="
vllm bench throughput --model "$MODEL" --dtype bfloat16 --tensor-parallel-size $TP \
    --input-len 512 --output-len 256 --num-prompts 128 --max-model-len 2048 --max-num-seqs 128 2>&1 | tee /tmp/bench-tp.log | grep -E '^Throughput|Error|OOM'
echo "=== single user latency (batch 1, 128 in / 256 out) ==="
vllm bench latency --model "$MODEL" --dtype bfloat16 --tensor-parallel-size $TP --batch-size 1 \
    --input-len 128 --output-len 256 --num-iters-warmup 3 --num-iters 10 --max-model-len 2048 \
    --max-num-seqs 8 --gpu-memory-utilization 0.5 2>&1 | tee /tmp/bench-lat.log | grep -E 'Avg latency|Error|OOM' | \
    awk '/Avg latency/{printf "Avg latency: %.2f s  =>  ~%.0f tokens/s for one user\n",$3,256/$3} !/Avg latency/{print}'
