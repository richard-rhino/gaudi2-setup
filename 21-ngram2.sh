#!/usr/bin/env bash
until grep -q 'CHAIN4-DONE' ~/setup/20-chain4.out 2>/dev/null; do sleep 30; done
SPEC="--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"
echo "### 30B-A3B TP=8 ngram-spec maxseqs32"; TP=8 LAZY=1 TOOL_PARSER=hermes EXTRA="$SPEC" bash ~/setup/12-office-10users.sh Qwen/Qwen3-30B-A3B-Instruct-2507 131072
bash ~/setup/stop-vllm.sh; echo NGRAM2-DONE
