#!/usr/bin/env bash
# gpt-oss-120b failed at TP=8 (view/stride error); the plugin validates it at TP=4. Retry as 2 replicas x 4 cards after the sweep.
until grep -q 'SWEEP-DONE' ~/setup/14-sweep.out 2>/dev/null; do sleep 30; done
TP=4 EXTRA="--data-parallel-size 2" LAZY=1 TOOL_PARSER=openai bash ~/setup/12-office-10users.sh openai/gpt-oss-120b 131072
bash ~/setup/stop-vllm.sh; echo GPTOSS-DONE
