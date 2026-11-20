#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (bf16, 360 GB) on all 8 Gaudi2 cards with the qwen4_exp HPU port (ports/qwen4_exp).
# Text only; QSA attention runs dense (exact up to ~2k context, see FINDINGS 14).
set -euo pipefail
source ~/vllm-venv/bin/activate
export HF_HOME=/models/huggingface PT_HPU_LAZY_MODE=0 PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_ENGINE_READY_TIMEOUT_S=7200
MAXLEN=${MAXLEN:-32768}
vllm serve Qwen/Qwen3.8-Flash-Next --served-model-name qwen38fn \
  --tensor-parallel-size 8 --dtype bfloat16 --max-model-len "$MAXLEN" --max-num-seqs ${SEQS:-32} \
  --limit-mm-per-prompt '{"image":0,"video":0}' --no-enable-prefix-caching \
  --gpu-memory-utilization ${MEM:-0.85} --port 8000 2>&1 | tee ~/gaudi2-setup/logs/serve-qwen38fn-$(date +%m%d-%H%M).log
