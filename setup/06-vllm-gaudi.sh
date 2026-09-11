#!/usr/bin/env bash
# vLLM on Gaudi2 using Intel's PyTorch container + the vllm-gaudi plugin.
# Needs 02-gaudi.sh and 03-docker.sh done, and your user in the docker group.
# Run as yourself:  bash setup/06-vllm-gaudi.sh
set -euo pipefail
IMG=vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest
NAME=vllm-gaudi
mkdir -p ~/models ~/.cache/huggingface

docker pull "$IMG"
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" 2>/dev/null || true
  docker run -d --name "$NAME" --runtime=habana --restart unless-stopped --ulimit memlock=-1 --ulimit nofile=1048576 \
    -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --ipc=host --net=host \
    -v ~/models:/models -v ~/.cache/huggingface:/root/.cache/huggingface \
    "$IMG" sleep infinity
fi
docker cp "$(dirname "$0")/06-in-container.sh" "$NAME":/root/install-vllm.sh
docker exec "$NAME" bash /root/install-vllm.sh

echo
echo "=== vLLM container '$NAME' ready ==="
echo "Serve a model (8B on one card):"
echo "  docker exec -it $NAME bash -lc 'vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --max-model-len 8192 --port 8000'"
echo "70B across 8 cards: add  --tensor-parallel-size 8"
echo "Gated models need:  docker exec -it $NAME huggingface-cli login"
