#!/usr/bin/env bash
# gpt-oss-120b: TP=8 and 2x(TP=4) both failed; try the exact validated layout (plain TP=4) on 4 cards.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
until grep -q 'NGRAM2-DONE' "$ROOT"/results/21-ngram2.out 2>/dev/null; do sleep 30; done
echo "### gpt-oss-120b TP=4 (4 cards)"; CARDS=0,1,2,3 GPU_MEM=0.7 GRAPH_MEM=0.2 TP=4 LAZY=1 TOOL_PARSER=openai bash "$ROOT"/experiments/12-office-10users.sh openai/gpt-oss-120b 131072
bash "$ROOT"/setup/stop-vllm.sh; echo GPTOSS4-DONE
