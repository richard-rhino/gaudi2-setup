#!/usr/bin/env bash
# Move the HF model cache to the NVMe (/mnt/nvme) and recreate the vLLM containers to use it.
set -euo pipefail
IMG=vllm-gaudi-stable:snap
[[ -d /mnt/nvme/huggingface/hub ]] || { echo "NVMe copy missing"; exit 1; }
docker rm -f vllm-gaudi-stable vllm-gaudi-gemma 2>/dev/null || true
if [[ ! -L ~/.cache/huggingface ]]; then
  mv ~/.cache/huggingface ~/.cache/huggingface.sata-old
  ln -s /mnt/nvme/huggingface ~/.cache/huggingface
fi
[[ -L ~/models ]] || { rsync -a ~/models/ /mnt/nvme/models/ && rm -rf ~/models && ln -s /mnt/nvme/models ~/models; }
mk() { docker run -d --name "$1" --runtime=habana --restart unless-stopped --ulimit memlock=-1 --ulimit nofile=1048576 \
  -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host \
  -v /mnt/nvme/models:/models -v /mnt/nvme/huggingface:/root/.cache/huggingface "$2" sleep infinity >/dev/null; }
mk vllm-gaudi-stable $IMG
mk vllm-gaudi-gemma $IMG
docker exec vllm-gaudi-gemma pip install -q "transformers==5.14.1" 2>&1 | grep -vE 'WARNING|resolver|requires|^$' || true
docker cp "$(dirname "$0")/vllm-serve.sh" vllm-gaudi-stable:/opt/vllm-serve.sh; docker cp "$(dirname "$0")/vllm-serve.sh" vllm-gaudi-gemma:/opt/vllm-serve.sh
docker ps --format '{{.Names}} {{.Status}}'
echo "Old SATA copy kept at ~/.cache/huggingface.sata-old (delete after confirming models load from NVMe)."
