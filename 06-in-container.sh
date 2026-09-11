#!/usr/bin/env bash
# Runs INSIDE the Intel Gaudi PyTorch container. Installs vLLM + vllm-gaudi plugin per
# NOTE: sources live in /opt/src, NOT /root: the Intel image has /root on PYTHONPATH and a bare
# /root/vllm checkout would shadow the installed package (ImportError: cannot import SamplingParams).
# https://github.com/vllm-project/vllm-gaudi README.
set -euo pipefail
mkdir -p /opt/src && cd /opt/src
[ -d vllm-gaudi ] || git clone https://github.com/vllm-project/vllm-gaudi
[ -d vllm ]       || git clone https://github.com/vllm-project/vllm
# VLLM_GAUDI_REF: a release tag (e.g. v0.26.0, validated for Gaudi 1.24.1) or "main"
VLLM_GAUDI_REF="${VLLM_GAUDI_REF:-v0.26.0}"
cd /opt/src/vllm-gaudi && git fetch -q --tags origin && git fetch -q origin vllm/last-good-commit-for-vllm-gaudi
if [[ "$VLLM_GAUDI_REF" == "main" ]]; then
  git checkout -q main && git pull -q
  VLLM_COMMIT=$(git show origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT)
else
  git -c advice.detachedHead=false checkout -q "$VLLM_GAUDI_REF"
  VLLM_COMMIT=$(git show "$VLLM_GAUDI_REF:VLLM_STABLE_COMMIT")
fi
echo "Using vLLM commit $VLLM_COMMIT"
cd /opt/src/vllm && git -c advice.detachedHead=false checkout -q "$VLLM_COMMIT"
pip install -r <(sed '/^torch/d' requirements/build/cuda.txt)
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
cd /opt/src/vllm-gaudi && pip install -e .
# main branch on 2026-09-09 needed transformers<5.17 (PixtralRotaryEmbedding removed in 5.17.0)
pip install "transformers==${TRANSFORMERS_PIN:-5.16.1}"
TORCH_VERSION=$(python3 -c "import re, torch; print(re.match(r'(\d+\.\d+\.\d+)', torch.__version__).group(1))")
pip install --no-deps "torchaudio==$TORCH_VERSION" --extra-index-url https://download.pytorch.org/whl/cpu
python3 -c "import vllm_gaudi, importlib.metadata as m; print('vllm', m.version('vllm'), '+ vllm_gaudi', m.version('vllm-gaudi'), 'OK')"
