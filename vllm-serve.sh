#!/usr/bin/env bash
# Runs INSIDE the container. Usage: /opt/vllm-serve.sh <model> [extra vllm args...]
# Examples:
#   /opt/vllm-serve.sh Qwen/Qwen2.5-0.5B-Instruct
#   HABANA_VISIBLE_DEVICES=all /opt/vllm-serve.sh meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 8
MODEL="${1:?model name required}"; shift || true
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}"
export HABANA_VISIBLE_DEVICES="${HABANA_VISIBLE_DEVICES:-0}"
cd /models   # never run from a directory that contains a 'vllm' checkout
exec vllm serve "$MODEL" --dtype bfloat16 --host 0.0.0.0 --port "${PORT:-8000}" "$@"
