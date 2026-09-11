#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="$ROOT"/experiments/12-office-10users.sh
echo "### 30B-A3B 2x(TP=4)";  GPU_MEM=0.7 GRAPH_MEM=0.2 TP=4 EXTRA="--data-parallel-size 2" LAZY=1 TOOL_PARSER=hermes bash $S Qwen/Qwen3-30B-A3B-Instruct-2507 131072
echo "### 30B-A3B 4x(TP=2)";  GPU_MEM=0.6 GRAPH_MEM=0.25 TP=2 EXTRA="--data-parallel-size 4" LAZY=1 TOOL_PARSER=hermes bash $S Qwen/Qwen3-30B-A3B-Instruct-2507 131072
echo "### gpt-oss-120b 2x(TP=4)"; GPU_MEM=0.7 GRAPH_MEM=0.2 TP=4 EXTRA="--data-parallel-size 2" LAZY=1 TOOL_PARSER=openai bash $S openai/gpt-oss-120b 131072
echo "### Coder-Next TP=8 eager 32k"; TP=8 LAZY=0 TOOL_PARSER=qwen3_coder bash $S Qwen/Qwen3-Coder-Next 32768
bash "$ROOT"/setup/stop-vllm.sh; echo CHAIN2-DONE
