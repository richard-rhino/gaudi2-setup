#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="$ROOT"/experiments/12-office-10users.sh
SPEC="--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"
echo "### 30B-A3B TP=8 ngram-spec"; MAX_SEQS=16 TP=8 LAZY=1 TOOL_PARSER=hermes EXTRA="$SPEC" bash $S Qwen/Qwen3-30B-A3B-Instruct-2507 131072
echo "### gpt-oss-120b 2x(TP=4)";  GPU_MEM=0.7 GRAPH_MEM=0.2 TP=4 EXTRA="--data-parallel-size 2" LAZY=1 TOOL_PARSER=openai bash $S openai/gpt-oss-120b 131072
echo "### Coder-Next TP=8 eager 32k"; TP=8 LAZY=0 TOOL_PARSER=qwen3_coder bash $S Qwen/Qwen3-Coder-Next 32768
bash "$ROOT"/setup/stop-vllm.sh; echo CHAIN4-DONE
