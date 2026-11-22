#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (bf16, 360 GB) on all 8 Gaudi2 cards with the qwen4_exp HPU port (ports/qwen4_exp).
# Text only; QSA attention runs dense (exact up to ~2k context, see FINDINGS 14).
set -euo pipefail
source ~/vllm-venv/bin/activate
export HF_HOME=/models/huggingface PT_HPU_LAZY_MODE=0 PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_ENGINE_READY_TIMEOUT_S=14400
# Full warmup = 90 prompt + 75 decode shapes at ~63 s each (~3 h). SKIP=1 compiles each shape on first use instead.
[ "${SKIP:-1}" = 1 ] && export VLLM_SKIP_WARMUP=true
MAXLEN=${MAXLEN:-32768}
# Small decode batches: compute only the routed experts instead of the fused op over all 512 (FINDINGS 18).
export VLLM_HPU_MOE_GATHER_BF16=${MOE_GATHER:-1}
# Pin rank i to module i: consecutive PP hops never use the flapping port-9 pairs (k <-> k+4), see setup-ubuntu2404/NIC-LINK-FLAPPING.md
export VLLM_HPU_MODULE_ORDER=${MODULE_ORDER-0,1,2,3,4,5,6,7}
# TP x PP must be 8. PP=8 TP=1 avoids per-layer all-reduces (the internal NIC links stall them, FINDINGS 18).
TP=${TP:-8}; PP=${PP:-1}
# PROFILE=<dir>: enable the torch profiler (CPU + HPU activity); drive it with POST /start_profile and /stop_profile
PROF=(); [ -n "${PROFILE:-}" ] && { export VLLM_TORCH_PROFILER_DIR="$PROFILE"
  PROF=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE\"}"); }
# MTP=k: k drafted tokens per step (patches 0007-0025). The HPU runner has no async scheduling with spec decode.
# WARNING: EXPERIMENTS ONLY. Needs 0019 for temperature > 0 and logprobs. PP=8: faster for several users, slower for one sampled user.
# Needs 0022: before it, greedy requests batched with sampled ones silently get wrong tokens (FINDINGS 21).
# Needs 0023 for structured output, 0024 for PP (FINDINGS 21).
SPEC=(); [ "${MTP:-0}" -gt 0 ] && SPEC=(--no-async-scheduling
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
vllm serve Qwen/Qwen3.8-Flash-Next --served-model-name qwen38fn \
  --tensor-parallel-size $TP --pipeline-parallel-size $PP --dtype bfloat16 --max-model-len "$MAXLEN" --max-num-seqs ${SEQS:-32} \
  --limit-mm-per-prompt '{"image":0,"video":0}' --no-enable-prefix-caching \
  --gpu-memory-utilization ${MEM:-0.85} --port 8000 "${PROF[@]}" "${SPEC[@]}" 2>&1 | tee ~/gaudi2-setup/logs/serve-qwen38fn-tp${TP}pp${PP}-$(date +%m%d-%H%M).log
