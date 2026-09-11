#!/usr/bin/env bash
# Chain the 10-user test over the small-active-parameter candidates. Results: ~/setup/office10-*.md
S=~/setup/12-office-10users.sh
TP=8 LAZY=1 TOOL_PARSER=hermes      bash $S Qwen/Qwen3-30B-A3B-Instruct-2507 131072
TP=8 LAZY=1 TOOL_PARSER=openai      bash $S openai/gpt-oss-120b 131072
TP=8 LAZY=0 TOOL_PARSER=qwen3_coder bash $S Qwen/Qwen3-Coder-Next 131072
bash ~/setup/stop-vllm.sh
echo ALL-DONE
