#!/usr/bin/env bash
# Qwen3-Coder-Next (GDN, eager only) warmup at 128k projected 19 h. Retry at 32k context after the other runs.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
until grep -q 'GPTOSS-DONE' "$ROOT"/results/15-gptoss.out 2>/dev/null; do sleep 30; done
TP=8 LAZY=0 TOOL_PARSER=qwen3_coder bash "$ROOT"/experiments/12-office-10users.sh Qwen/Qwen3-Coder-Next 32768
bash "$ROOT"/setup/stop-vllm.sh; echo CODERNEXT-DONE
