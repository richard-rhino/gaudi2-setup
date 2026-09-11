#!/usr/bin/env bash
# llama.cpp, CPU build. Gaudi has no llama.cpp backend, so this uses the 2x Xeon 6338 (64 cores) + 125 GB RAM.
# Run as yourself:  bash setup/05-llama-cpp.sh
set -euo pipefail
cd ~
[[ -d llama.cpp ]] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
mkdir -p ~/models
echo
echo "=== llama.cpp built: ~/llama.cpp/build/bin ==="
echo "Try (downloads ~5 GB GGUF from Hugging Face on first use):"
echo "  ~/llama.cpp/build/bin/llama-cli -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M -t 64 -c 8192 -p 'Hello'"
echo "  ~/llama.cpp/build/bin/llama-server -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M -t 64 -c 8192 --host 0.0.0.0 --port 8080"
