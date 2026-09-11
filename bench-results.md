# Gaudi2 vLLM benchmarks (2026-09-10)

Host: 8x Gaudi2 (HL-225, 96 GB each), vLLM v0.26.0 + vllm-gaudi v0.26.0, bf16, untuned defaults.
Batched = 128 concurrent prompts, 512 in / 256 out. Single user = batch 1, 128 in / 256 out.

| Model | Released | Cards (TP) | Mode | Batched output tok/s | Total tok/s | Single-user tok/s | Warmup |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.8-27B | 2026-08-14 | 1 | eager | 264 | 2,376 | ~37 | 3 min |
| Qwen/Qwen3.8-27B | 2026-08-14 | 1 | lazy, no HPU graphs | 131 | 1,179 | - | 0 |
| Qwen/Qwen3.6-35B-A3B | 2026-04-16 | 1 | eager | 212 | 1,904 | ~30 | 17 min |
| google/gemma-4-31b-it | 2026-04-02 | 1 | lazy (transformers 5.14.1) | 240 | 2,162 | ~32 | 2.5 min |
| Qwen/Qwen2.5-32B-Instruct | 2024 | 4 | lazy | 980 | 8,817 | ~34 | 2 min |
| Qwen/Qwen2.5-7B-Instruct | 2024 | 1 | lazy | 1,701 | 15,309 | ~128 | 2 min |
| Qwen/Qwen2.5-0.5B-Instruct | 2024 | 1 | lazy | 3,736 | 33,627 | - | 2 min |

Notes
- Qwen3.5/3.6/3.8 (Gated DeltaNet hybrid) crash in lazy mode with HPU graphs ("Neither storage attached to input
  tensor"); eager mode (PT_HPU_LAZY_MODE=0) works but is slower than the lazy+graphs path the 2024 models use.
- Gemma 4 needs transformers<=5.14.1 with vLLM 0.26 (per-layer head_dim API changed in 5.15); container `vllm-gaudi-gemma`.
- Not runnable today: Qwen3.8-Flash-Next / GLM-5.3-Flash (new architectures not in vLLM 0.26 / plugin),
  DeepSeek-V4-Flash (FP4 experts), GLM-5.3 753B / Kimi K3 2.8T / Qwen3.8-2.4T (exceed 768 GB HBM), MiniMax-M3 (bf16 854 GB; MXFP8 unsupported by plugin).
| Qwen/Qwen3.6-35B-A3B | 1 (1) | 211.52 | 1903.66 | 30 | RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {' |
