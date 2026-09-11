#!/usr/bin/env bash
# n-gram speculative decoding retry with quoting that survives the docker exec string.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
until grep -q 'CHAIN3-DONE' "$ROOT"/results/18-chain3.out 2>/dev/null; do sleep 30; done
SPEC="--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"
echo "### 30B-A3B TP=8 ngram-spec (retry)"; MAX_SEQS=16 TP=8 LAZY=1 TOOL_PARSER=hermes EXTRA="$SPEC" bash "$ROOT"/experiments/12-office-10users.sh Qwen/Qwen3-30B-A3B-Instruct-2507 131072
bash "$ROOT"/setup/stop-vllm.sh; echo NGRAM-DONE
