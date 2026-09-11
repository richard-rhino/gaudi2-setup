#!/usr/bin/env bash
# Parallelism sweep for Qwen3-30B-A3B: replicas of fewer cards may give higher per-user tok/s than TP=8.
S=~/setup/12-office-10users.sh
until grep -q 'ALL-DONE' ~/setup/13-candidates.out 2>/dev/null; do sleep 30; done
TP=4 EXTRA="--data-parallel-size 2" LAZY=1 TOOL_PARSER=hermes bash $S Qwen/Qwen3-30B-A3B-Instruct-2507 131072
TP=2 EXTRA="--data-parallel-size 4" LAZY=1 TOOL_PARSER=hermes bash $S Qwen/Qwen3-30B-A3B-Instruct-2507 131072
bash ~/setup/stop-vllm.sh; echo SWEEP-DONE
