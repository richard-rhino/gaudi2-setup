#!/usr/bin/env bash
# User-level stack: Intel PyTorch for Gaudi (venv) + vLLM (pinned) + vllm-gaudi plugin.  bash setup-ubuntu2404/03-user-stack.sh
# Prereqs: 01-root-setup.sh, 02-rebuild-driver.sh, reboot. Gaudi env loaded (~/.zshenv sources /etc/profile.d/habanalabs*.sh).
set -euo pipefail
GAUDI_VER=1.24.2
VENV_NAME=vllm-venv          # NOTE: the installer prepends $HOME to HABANALABS_VIRTUAL_DIR, so give a RELATIVE name.

mkdir -p ~/setup/installer && cd ~/setup/installer
[[ -f habanalabs-installer.sh ]] || wget -nv "https://vault.habana.ai/artifactory/gaudi-installer/$GAUDI_VER/habanalabs-installer.sh"
chmod +x habanalabs-installer.sh
HABANALABS_VIRTUAL_DIR=$VENV_NAME ./habanalabs-installer.sh install -t pytorch --venv -y
source ~/$VENV_NAME/bin/activate

mkdir -p ~/src && cd ~/src
[[ -d vllm-gaudi ]] || git clone https://github.com/vllm-project/vllm-gaudi
cd vllm-gaudi && git fetch origin vllm/last-good-commit-for-vllm-gaudi
VLLM_COMMIT=$(git show origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT)
cd ~/src && [[ -d vllm ]] || git clone https://github.com/vllm-project/vllm
cd vllm && git checkout "$VLLM_COMMIT"
pip install -r <(sed '/^torch/d' requirements/build/cuda.txt)
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
cd ~/src/vllm-gaudi && pip install -e .
pip install --no-deps torchaudio==2.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
python -c "import torch, vllm, vllm_gaudi, habana_frameworks.torch.hpu as h; print('torch', torch.__version__, 'vllm', vllm.__version__, 'hpus', h.device_count())"
