# Intel Gaudi2 LLM server: findings log

Host: Supermicro, 2x Xeon Gold 6338 (128 threads), 125 GB RAM, 8x Intel Gaudi2 HL-225 (96 GB HBM each, 768 GB total),
1.7 TB SATA SSD (OS) + 2.9 TB NVMe (models). OS: Debian 12, kernel 6.1. Dates: 2026-09-09 to 2026-09-10.
All scripts, logs and result tables live in `~/setup/`. Memory notes for future sessions are in `~/.claude/.../memory/`.

---

## 1. Base system and shell
- Fresh Debian had no compilers, git, or zsh. `01-base.sh` installs the toolchain, kernel headers, zsh, fzf/ripgrep/bat/btop,
  and every library the Intel packages depend on. `04-zsh.sh` sets up oh-my-zsh + autosuggestions + syntax highlighting + powerlevel10k.
- llama.cpp was built (`05-llama-cpp.sh`) but **llama.cpp has no Gaudi backend**; it only runs on the CPUs here. The Gaudi path is vLLM.

## 2. Gaudi driver on Debian 12 (unsupported OS)
- Intel supports Ubuntu 22.04/24.04 and RHEL 9 only. Their installer refuses Debian 12 outright.
- Workaround that works: point apt at Intel's Ubuntu 22.04 (`jammy`) repo, which contains only `habanalabs-*` packages
  with ordinary Debian dependencies. Packages are pinned with `apt-mark hold`. (`02-gaudi.sh`)
- **Bug 1: DKMS build failed.** Intel's `compat/scripts/generate_flags.sh` mis-parses Debian's kernel-headers layout
  (the arch headers Makefile `include`s the common headers by absolute path; the script assumed relative), so every
  kernel-feature probe failed and it tried to re-implement APIs the 6.1 kernel already has. A 10-line patch fixes it
  (`generate_flags.sh.patched`, applied automatically by `02-gaudi.sh`). Must be re-applied after any driver upgrade.
- After that, all 8 cards showed in `hl-smi`, firmware already current (1.24.0-fw-62.6.2). Driver autoloads at boot.
- **Bug 2: silent wrong answers.** With the driver loaded, small tensor ops were exact but any real model produced
  gibberish, and large gathers / long-K matmuls returned garbage that changed run to run. Root cause: Debian boots the
  host IOMMU in translated (DMA-remapping) mode; Intel requires passthrough. Adding `intel_iommu=on iommu=pt` to GRUB
  and rebooting fixed it completely (every HPU-vs-CPU test exact and deterministic afterwards). This one cost ~3 hours.
- Also configured: hugepages per Intel's formula (~27 GB), `--ulimit memlock=-1` on containers (HCCL init failed for a
  second process without it), core dumps disabled (a crashed run left a 9.6 GB core file).
- Verdict: **Debian works, but Ubuntu 22.04 would have avoided both bugs.** Everything is scripted so a re-install is quick.

## 3. Software stack
- Docker CE + Habana container runtime (`03-docker.sh`). Intel's `pytorch-installer-2.11.0` image (Gaudi 1.24.1).
- vLLM v0.26.0 built for the "empty" device + `vllm-gaudi` plugin v0.26.0 (Intel's validated pair). Container `vllm-gaudi-stable`.
  transformers pinned to 5.16.1 (5.17 removed a class vLLM 0.26 imports). Gemma 4 needs transformers 5.14.1 -> container `vllm-gaudi-gemma`.
- Sources must live outside `/root` inside the image: `/root` is on PYTHONPATH and a bare `vllm/` checkout shadows the package.
- A `vllm-gaudi-main` container (plugin main @ 2026-09-09) exists for testing fixes; it did not fix the Qwen3.x crash (see 6).

## 4. First benchmarks (single model, batched throughput, 2024 models)
| Model | Cards | Batched output tok/s (128 prompts, 512 in/256 out) | Single user tok/s |
|---|---|---|---|
| Qwen2.5-7B | 1 | 1,700 | ~128 |
| Qwen2.5-32B | 4 (TP) | 980 | ~34 |
| Qwen2.5-0.5B | 1 | 3,700 | - |
Model loading from the SATA SSD ran at 350 MB/s (2.5 min for a 27B). Moved the model cache to the NVMe (3.4 GB/s read); `~/.cache/huggingface` and `~/models` are symlinks to `/mnt/nvme`.

## 5. Which recent (Mar-Sep 2026) models can run here
| Model | Status |
|---|---|
| Qwen3.8-27B (Aug 14) | Runs (eager mode only, see 6) |
| Qwen3.6-27B / 35B-A3B (Apr) | Run (eager mode only) |
| Gemma-4-31B (Apr 2) | Runs (fast path), needs transformers 5.14.1 |
| Qwen3-Coder-Next (Feb 3) | Downloaded, test pending (Qwen3-Next arch, plugin lists it as supported) |
| Qwen3.8-Flash-Next, GLM-5.3-Flash | New architecture classes not in vLLM 0.26 / plugin |
| DeepSeek-V4-Flash | FP4 experts; Gaudi2 has no FP4 compute, no bf16 checkpoint |
| GLM-5.3 (753B), Kimi K3 (2.8T), Qwen3.8-2.4T | Exceed 768 GB HBM |
| MiniMax-M3 (428B) | bf16 is 854 GB; the MXFP8 build uses a quant format the plugin rejects |

## 6. Single-card results for the 2026 models (batched, 1 card)
| Model | Mode | Batched output tok/s | Single user tok/s |
|---|---|---|---|
| Qwen3.8-27B | eager | 264 | ~37 |
| Qwen3.6-35B-A3B | eager | 212 | ~30 (17 min warmup) |
| Gemma-4-31B | lazy+graphs | 240 | ~32 |
Finding: the Qwen3.5/3.6/3.8 family (Gated DeltaNet hybrid attention) **crashes on the fast lazy+HPU-graph path**
("Neither storage attached to input tensor" in graph replay) on plugin v0.26.0 and on main. Eager mode (which already uses
torch.compile) works but is ~3-4x slower per card than the fast path. `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false` avoids the
crash but then warmup runs out of device memory even at TP=2. Dead end for now; re-test on the next plugin release.

## 7. 8-card serving test ("office load"), Qwen3-235B-A22B
Served through the OpenAI-compatible API, 32k context, load generated by `vllm bench serve` (`11-office-load.sh`).
- FP8 checkpoint cannot be split 8 ways by tensor parallel (expert width 1536/8 = 192, not divisible by the 128-wide quant
  blocks); data-parallel replicas do not help because vLLM still slices experts across all workers. `--enable-expert-parallel`
  (whole experts per card, 16 of 128 each) is required. Startup 32 min (19 min warmup).
| Scenario | Users | In/out tokens | Output tok/s (aggregate) | TTFT p50/p99 | Per-token latency |
|---|---|---|---|---|---|
| burst | 64 at once | 1k/256 | 322 | 7.1 s / 12.6 s | 149 ms (~7 tok/s per user) |
| sustained | 32 | 2k/512 | 212 | 0.7 s / 10 s | 136 ms |
| sustained | 64 | 2k/512 | 363 | 0.9 s / 20 s | 151 ms |
| long ctx | 8 | 12k/512 | 45 | 4 s / 29 s | 153 ms |
- bf16 checkpoint (plain TP=8): 105 ms per token (~10 tok/s per user), 433 out tok/s at 64-user burst. **bf16 is ~40% faster
  than FP8 on Gaudi2** with this plugin. Either way a 22B-active model gives ~7-10 tok/s per user: fine for batch, poor for chat.

## 8. Requirement: >=10 users, >=30 tok/s each, 128k context, chat (Open WebUI) + agentic (OpenCode)
Test: `12-office-10users.sh` (server with 128k max_model_len, tool-call parser enabled; scenarios below). "Per-user tok/s" = 1000 / median time-per-output-token.
| Model | Cards | chat 10 users (1k/512) | chat 16 users | agentic 10 users (32k/1k) | mixed 16 (8k/768) | 4 users @100k | TTFT chat p50 | Startup |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (eager) | 8 TP | 21.6 | 21.9 | 17.3 | 18.4 | 13.9 | 0.3 s | 12 min |
| Qwen3-30B-A3B-2507 (lazy) | 8 TP | 28.9 | 27.6 | 19.9 | 23.8 | 11.9 | 0.2 s | 11 min |
| gpt-oss-120b (MXFP4) | 8 or 4 TP | FAILED in every layout (TP=8, 2x TP=4, plain TP=4): the MXFP4 expert path crashes at the first warmup step on Gaudi2 ("view size is not compatible with input tensor size and stride"). Plugin validates it on Gaudi 3 only. The bf16 re-export (lmsys/gpt-oss-120b-bf16) fails with the same error, so the fault is in the gpt-oss attention path on HPU, not the 4-bit format. gpt-oss is out on Gaudi2 with plugin v0.26.0. | | | | | | |
| Qwen3-Coder-Next (eager) | 8 TP | ABORTED twice: eager-mode warmup projected 19 h at 128k and 14 h at 32k (~7 min per prompt bucket). Same GDN family as Qwen3.8; not viable until the plugin's fast path works for it. | | | | | | |
| Qwen3-30B-A3B, 2x(TP=4) replicas (lazy) | 8 | 17.6 | 17.2 | - | - | - | 0.3 s | 13 min |
| Qwen3-30B-A3B TP=8, max-num-seqs 16 | 8 | 28.2 | 29.8 | (see office10-*.md) | | | 0.2 s | 11 min |
| Qwen3-30B-A3B TP=8 + n-gram speculative decoding | 8 | FAILED in warmup both at max-num-seqs 16 and 32 ("IndexError: index N is out of bounds for axis 0 with size N", N = max-num-seqs): off-by-one in the plugin's experimental spec-decode warmup. Dead end on v0.26.0. | | | | | | |
| Qwen3-30B-A3B, 4x(TP=2) replicas | 8 | skipped: 2x(TP=4) was already slower per user than TP=8, so fewer cards per replica is the wrong direction here | | | | | | |
Observations so far:
- Qwen3.8-27B: no. 22 tok/s flat, and it is stuck on the slow path (section 6).
- Replica layouts: first attempts OOMed in warmup at the default memory split (gpu_mem 0.7 + graph_mem 0.2 fixes it) and then
  hit vLLM's 600 s engine-ready timeout (VLLM_ENGINE_READY_TIMEOUT_S=7200 fixes it). Result: **fewer cards per replica is slower
  per user** (17.6 vs 28.9 tok/s). Per-user decode speed on this stack scales with tensor-parallel width, not replica count.
- Qwen3-30B-A3B: almost. 29 tok/s at 10 users. Long prompts (32k+) cost 4-5 s to first token and pull per-user speed to ~20.
  Prefill (prompt processing) is the weak spot of this stack; decode scales well with users.
- Memory is never the limit: KV caches of 1-4 million tokens on 8 cards; 128k contexts fit easily.

## 9. Operational pitfalls (each cost real time)
- Killing `vllm serve` can orphan `VLLM::Worker_TP*` processes that keep the cards' memory pinned (82 GB/card "in use"); the
  next launch fails with "Device acquire failed". Use `stop-vllm.sh`, which kills workers in every container and waits for hl-smi.
- After streaming hundreds of GB of weights, the kernel's low DMA zone fragments and the driver's coherent-DMA allocation
  fails ("ctx_init failed", "Failed to create context -12", userspace sees "Device not found") even with idle cards.
  `drop_caches` + `compact_memory` fixes it; `stop-vllm.sh` now does this automatically.
- Warmup (graph pre-compilation for every batch/context bucket) is 10-30 min per server start at 128k context. Plan for it.
- `vllm bench latency` and warmup OOM at the default memory fraction on some models; `--gpu-memory-utilization 0.5` fixes it.
- Multi-replica (`--data-parallel-size`) servers run several API-server processes that give up after 600 s if the engines are
  still warming up ("Timed out waiting for engine core processes to start"). Set `VLLM_ENGINE_READY_TIMEOUT_S=7200`.

## 10. Recommendation (final for this plugin version)
**Qwen3-30B-A3B-Instruct-2507, 8-way tensor parallel, 128k context.** Measured 28-30 tok/s per user at 10-16 concurrent chat
users with 0.2 s to first token, ~20-24 tok/s on 8-32k agentic prompts, and it degrades gracefully to 64+ users. It is the
only tested model that is both fast enough and stable on the plugin's fast path. Everything newer either runs 2-3x slower
(Qwen3.6/3.8, Coder-Next: eager mode only, hours of warmup) or does not run at all (gpt-oss, DeepSeek V4, GLM-5.3, Kimi K3).
What was tried to go faster and did not help: fewer cards per replica (slower), smaller batch limit (no change),
n-gram speculative decoding (plugin bug), FP8 (slower than bf16 on Gaudi2).
The honest gap: a 32k-token agentic prompt waits 4-5 s for the first token, and 100k prompts ~45 s. Prefill is this stack's
weakness; decode is fine. Re-evaluate when vllm-gaudi ships a fix for the Qwen3.x hybrid-attention fast path, which would
make Qwen3.8-27B (Aug 2026) the obvious upgrade at roughly double its current speed.

## 11. Deployment (2026-09-10 evening)
`~/deploy/docker-compose.yml` runs the chosen model as a service: vLLM on :8000 (OpenAI-compatible, API key in `.env`)
across all 8 cards with a 128k context and tool calling enabled, plus Open WebUI on :3000. `~/deploy/opencode.json` is the
OpenCode provider config. Model switching is a one-line `.env` change (every restart re-pays the ~11 min warmup).
Chosen model: **Qwen3-30B-A3B-Instruct-2507** (Qwen3 2507 refresh, native 262k context) at TP=8, the best measured
per-user speed of anything that runs on the fast path. gpt-oss-120b-bf16 was tested and fails (section 8). The service was restarted with Qwen3-30B-A3B; API, tool calling, and Open WebUI verified end to end.

## 12. Model swap to Qwen3.8-27B (user decision, 2026-09-10 23:00)
The user chose Qwen3.8-27B over the faster 30B-A3B ("doesn't meet the requirements but it'll work well enough"): a current
(Aug 2026) dense model with thinking, at ~22 tok/s per user. Deployment changes: eager mode (LAZY_MODE=0), tool parser
`qwen3_xml` (Qwen3.5+ XML `<function=...>` format, verified from the chat template), `--reasoning-parser qwen3` so thinking
goes to the reasoning field. The compose file now takes all of these from `.env`.
Also learned while the 30B was live: Open WebUI injects its built-in knowledge tools (search/query/view_knowledge_*) into
every chat; Qwen3-30B-A3B called them ~300 times in a loop for a plain story prompt. Disable them per model or globally.
