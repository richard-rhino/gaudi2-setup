# Intel Gaudi2 LLM server: findings log

Host: Supermicro, 2x Xeon Gold 6338 (128 threads), 125 GB RAM, 8x Intel Gaudi2 HL-225 (96 GB HBM each, 768 GB total),
1.7 TB SATA SSD (OS) + 2.9 TB NVMe (models). OS: Debian 12, kernel 6.1. Dates: 2026-09-09 to 2026-09-10.
Scripts are in `setup/` and `experiments/`, result tables in `results/`, raw logs in `logs/`.

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
| Qwen3-30B-A3B TP=8, max-num-seqs 16 | 8 | 28.2 | 29.8 | (see results/office10-*.md) | | | 0.2 s | 11 min |
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

---
# Second server (Ubuntu 24.04), 2026-11-20: porting Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash

## 13. Setup on Ubuntu 24.04 (supported OS)
Host: 8x Gaudi2 HL-225, 2x Xeon 8380 (160 threads), 503 GB RAM, 2.9 TB NVMe. Gaudi SW 1.24.2, Intel PyTorch 2.13,
vLLM 0.34.0rc0 (commit pinned by vllm-gaudi) + vllm-gaudi main (0.27.0rc1.dev199), all native in a venv (no Docker).
Scripts and the pitfall list are in `setup-ubuntu2404/README.md`. On a supported OS the install is much simpler than on Debian
(no driver patch), but it still cost time: a kernel bump left the new kernel on the in-tree driver, the installer
mangles absolute venv paths, and **Ubuntu's zsh does not read /etc/profile.d**, so without `zshenv` every HPU op fails
with "Graph compile failed".
Smoke test (Qwen2.5-0.5B, 1 card, same script as section 4): **7,225 output tok/s batched** (Debian/plugin 0.26: 3,700),
182 tok/s single user.

## 14. Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) on Gaudi2
Architecture: 48 layers, 36 Gated DeltaNet + 12 "QSA" (indexed sparse) attention, 4-stream hyper-connections,
a 102 GB hashed n-gram embedding (PLE) at layer 2, 512 experts top-10 (+ shared), MTP, vision tower. 360 GB bf16.
Upstream vLLM has it only as `nvidia/` (Triton, CuTe) and `amd/` (Triton, aiter) backends; on the HPU (an out-of-tree
platform) it falls through to the NVIDIA code and dies at the first kernel. The plugin does not support it.

**Approach.** A self-contained text-only port (`ports/qwen4_exp/qwen4_exp.py`, 491 lines, plus ~20 lines of plugin
integration; `vllm-gaudi-patches/`) built from generic vLLM layers + plugin HPU ops, with the model-specific parts
written in plain torch from the HF reference (`transformers.models.qwen4_exp`, pure torch, which made this tractable):
| Component | How |
|---|---|
| Gated DeltaNet (36 layers) | reused the plugin's `HPUGatedDeltaNetAttention` (sigmoid output gate) |
| QSA attention (12 layers) | **dense** gated attention (`Qwen3NextAttention`). The indexer keeps every visible token while a query sees <= 512 complete 4-token blocks, so dense is exact up to ~2k context and a superset beyond. |
| MoE 512x top-10 + shared | reused `Qwen3NextSparseMoeBlock` (HPU fused MoE) |
| Hyper-connections | ~30 lines of torch |
| PLE n-gram embedding | hashing on **CPU** (see below); 102 GB table vocab-sharded over TP ranks (12.8 GB/card at TP=8) |
| PLE dilated short conv + token context state | per-request state stored in a slot that **borrows the GDN state spec**, so the runner allocates/indexes it like a 37th GDN layer (no runner changes beyond naming). History packed as raw bf16 in the fp32 slot: fits up to TP=16. |

**Gaudi gaps found on the way** (repros in `ports/qwen4_exp/tests/`):
- **int64 arithmetic is silently wrong on HPU.** `t*m` (t<2^18, m~2^45, product <2^63), `bitwise_xor` and `remainder`
  return wrong values for 4096/4096 elements with no error; int64 `gather` fails to compile. The PLE n-gram hash needs
  exact int64, so a naive port would run and generate text while looking up garbage embedding rows. Fix: hash on the CPU.
- `torch.cummax`, `torch.linalg.solve_triangular`, `scatter` into bool tensors: "Graph compile failed". Replacements:
  log-step running max; an exact matmul-only inverse for unit-lower-triangular matrices ((I+N)^-1 = prod (I+(-N)^(2^k)),
  6 matmuls for 64-wide chunks); int8 scatter.
- Debugging: with the default async eager pipeline the error surfaces at a *later* op; `PT_HPU_EAGER_PIPELINE_ENABLE=0`
  gives the real one.

**Plugin integration bugs found:** transformers 5.8+ renames `full_attention` -> `indexed_attention`, so the plugin
counted **0 attention layers** for KV-cache planning; the HPU runner passes **2-D `[batch, seq]` input_ids** (models must
not assume flat tokens); models are built under a default-device=hpu context, so CPU-side tables need explicit
`device="cpu"`; the runner reads `model.config.architectures`, so the top-level config must be exposed; hybrid
architectures are an allow-list in the runner and platform.

**Validation** (tiny random-weight Qwen4Exp with every architecture feature, vs HF on CPU; details and tables in
`results-ubuntu2404/qwen4_exp/VALIDATION.md`): prefill and 32-step decode agree to bf16 noise (mean |dlogprob|
0.001-0.003) at 7-300 token prompts, TP=1 and TP=2, eager and torch.compile+HPU-graph modes. Dense-vs-sparse QSA costs
0.011-0.026 mean |dlogprob| once the context exceeds the indexer budget (measured with a 64-token budget).

**Status:** the real 360 GB checkpoint runs, first at TP=8 and now faster with pipeline parallelism (PP=8). See
section 18.

## 15. DeepSeek-V4.1-Flash-1017: feasibility and preparation
Much more work than Qwen: there is no transformers implementation of V4.1, and upstream vLLM's two backends (CUDA:
FlashMLA, DeepGEMM, TileLang, CuTe; ROCm: aiter, flydsl, Triton) share an abstract attention layer with direct
`torch.ops._C` and `torch.cuda` calls. It needs HPU backends for three new KV-cache types (sliding-window MLA,
compressed MLA, indexer; plus a circular-buffer compressor state the plugin cannot allocate), engram (two 98 GB
n-gram tables, again int64 hashing), Sinkhorn hyper-connections, sqrtsoftplus routing and FP4 experts.
- **FP4 experts on Gaudi2 (no FP4 hardware).** Measured on the real checkpoint (`ports/deepseek_v41/scale_spread.py`,
  483 sampled expert scale tensors, 1.5M rows): the E8M0 scale spread within a row is median 1, p99 2, **max 5**
  (log2). Converting to Gaudi2 E4M3 (max 240) with a per-row scale 2^(max-5) is exact while the spread is <= 13, so
  **FP4 -> per-channel FP8 is bit-exact for 100% of sampled rows** and can use the HPU's native FP8 MoE path.
  (The plugin's block-FP8 MoE path instead dequantizes every expert to bf16 on every forward.)
- **Memory:** FP8 experts 544 GB + attention ~20 GB + engram 197 GB = ~761 GB vs 768 GB HBM. Engram has to live in host
  RAM (only gathers), or the experts stay FP4-packed (272 GB) and are decoded on the fly.
- **Plan "Track A"**: run DeepSeek's own reference (`inference/model.py`, which already supports 8-way model parallel)
  on the HPUs with its 6 TileLang kernels replaced by torch (`ports/deepseek_v41/kernel_torch.py`: exact fp32 FP8/FP4
  grid rounding, no fp8 dtypes on the device, LUT byte decoders), validate against the Nvidia server, then measure.
  A vLLM port is the step after.
- A local LLM (DeepSeek-V4.1 on the Nvidia node) drafted the torch kernels (`kernel_helper.py`). Review found it
  mathematically right except **FP4 round-half ties** (argmin breaks ties downward: 0.75->0.5, 1.75->1.5, 3.5->3.0
  instead of RNE 1.0/2.0/4.0, which matters because power-of-two scales make exact ties common), plus fp32 GEMMs and
  an unbounded attention gather. The final kernels are mine; CPU validation of them (`test_kernels_cpu.py`) could not run
  tonight because of the crash.

## 15b. Using the LAN's Nvidia-hosted models as helpers (`setup-ubuntu2404/ask-llm`)
- DeepSeek-V4.1 drafted the torch kernel replacements (useful skeleton; one real bug in FP4 tie rounding, see 15).
- Qwen3.8-Flash-Next (the model being ported) reviewed the Qwen port against the HF reference
  (`results-ubuntu2404/qwen4_exp/review-by-local-qwen.md`): it found no math differences, which matches the numeric
  validation, but the review is noisy (it labels 20+ items "CRITICAL ISSUE" and then concludes "identical" for each,
  and it ran out of tokens). With thinking enabled it spent the whole 12k-token budget on reasoning and returned nothing.
  Net: good for drafts and second opinions, but every claim needed checking.

## 16. Driver crash (habanalabs 1.24.2 on kernel 6.8.0-151): everything stopped at 04:49
- Trigger: starting a vLLM run in lazy mode (`PT_HPU_LAZY_MODE=1`) on module 0 (bus 43:00.0).
- `dmesg`: `BUG: kernel NULL pointer dereference` in `hbl_ib_eqe_work_schd` [habanalabs_ib], a second Oops, then
  `watchdog: BUG: soft lockup - CPU#72 stuck` in `gaudi2_cn_phy_link_status_work -> do_fw_tuning` [habanalabs_cn]
  that never ends.
- The vLLM process is stuck in D state in `hbl_cn_eq_dispatcher_register_cq` holding 97.5 GB on that card, and every
  new process that calls `sched_setaffinity` (the HPU runtime, OpenMP init, even `taskset`) hangs in `affine_move_task`.
  So no new PyTorch process can start, CPU-only included. kill -9 does not help. **Only a reboot recovers.**
- Background: **internal NIC port 9** (scale-up link between cards; the external ports are 8, 22, 23) is DOWN on 5 of 8
  cards (19, 1a, 43, 44, b3:00.0) and was flapping ("Card N Port 9: link down/up") from ~15 min after boot. The crash is
  the IB driver handling a link event while a new HCCL context is being created. Card firmware is 1.24.0-fw-62.6.2
  (driver 1.24.2). Kernel 6.8.0-151 was pulled in by apt (`linux-headers-generic`); 6.8.0-53 also has the DKMS driver.
- Lazy mode is the legacy path (this plugin defaults to `PT_HPU_LAZY_MODE=0`, eager + torch.compile, which works).

## 17. After the reboot: firmware re-flash and the internal NIC links (2026-11-20)
- Re-flashed all 8 cards with `hl-fw-loader` (guide: `setup-ubuntu2404/FIRMWARE-UPDATE.md`). The package ships the
  same firmware the cards already had (`1.24.0-fw-62.6.2`), so the version did not change.
- 8-card 128 MB all-reduce went from 2-3 GB/s before the flash to 39-42 GB/s after it, and later 199 GB/s
  (`experiments-ubuntu2404/allreduce_latency.py`). **Internal port 9 keeps flapping on every card**, about 90 link
  up/down events per minute while serving.
- Masking port 9 (`nic_ports_mask=0xFFFDFF`) makes HCCL hang. Do not mask internal ports.
- Lazy mode (`PT_HPU_LAZY_MODE=1`) triggered the crash in section 16. Everything below uses eager + torch.compile.

## 18. Qwen3.8-Flash-Next on the real checkpoint: correctness and speed (2026-11-20/21)
Serve script: `experiments-ubuntu2404/qwen38fn-serve.sh` (`TP=`, `PP=`, `MOE_GATHER=`). Stop with
`setup-ubuntu2404/stop-vllm.sh` (SIGINT is ignored during warmup). Reference: deb-ai (Nvidia, same model, MTP
speculative decoding), light single-user requests only.

**Correctness** (`experiments-ubuntu2404/compare_vs_nvidia.py`, 4 prompts of 45-3414 chars): mean prompt-token
|dlogprob| against deb-ai is 0.02-0.18 in every configuration below. Greedy text agrees for 113-294 characters (the
code prompt for all 201). Divergences are near-ties ("Italy" vs "Australia" after "The capital of Japan is Tokyo.").
Results: `results-ubuntu2404/qwen38fn/compare_vs_deb-ai*.txt`.

**Speed, step by step** (single user, 256 output tokens, `single_user_speed.py`; 16 users = `vllm bench serve`
random 1024 in / 256 out, concurrency 16, after a warm pass):
| Configuration | Single-user decode | 16 users, aggregate | TPOT at 16 users |
|---|---|---|---|
| TP=8 | 2.6-3.4 tok/s | not measured (compile stalls) | ~200-390 ms ITL |
| PP=8, TP=1 | 9.6-14.9 tok/s | 36.5 tok/s | 428 ms |
| PP=8 + routed-expert gather MoE | **~27 tok/s** (14-27, see below) | **123 tok/s** (peak 169) | 120 ms |
| deb-ai (Nvidia, MTP spec decode) | 139-148 tok/s | not tested (shared box) | |

Why each step helped:
1. **TP=8 is bound by the network, not compute.** A device profile of 12 decode steps (needed a profiler fix, below)
   shows the matrix engines busy ~105 ms in total, i.e. ~10 ms of compute per step out of ~300 ms. The rest is waiting
   for the ~98 all-reduces per step. A decode-sized (5 KiB) 8-card all-reduce takes ~220 us at p50, but stalls for
   **4-16 s every 10-30 s** and hung for good after ~90 s (`experiments-ubuntu2404/allreduce_latency_trace.py`,
   `logs/allreduce-latency-trace-*.log`). That matches the flapping port 9 links (section 17).
2. **Pipeline parallelism avoids the all-reduces.** A chain of point-to-point sends through all 8 cards (one PP decode
   step) takes ~0.8 ms with no stall in 120 s (`p2p_latency_trace.py`). vllm-gaudi had no real PP path, so the port
   adds one: the model builds only its stage's layers (`make_layers`), the HPU runner does a blocking send/recv of the
   hyper-connection hidden state around each forward pass, and only the last stage samples (the scheduler sends the
   tokens back to the other stages). The 102 GB PLE table does not fit on one card, so with PP it lives in **host RAM**:
   rows are gathered on the CPU, where the n-gram ids are already hashed. KV cache grows from 3.3M to 10.5M tokens.
3. **The fused MoE kernel reads all 512 experts.** `torch.ops.hpu.mixture_of_experts` costs 2.1-2.8 ms per layer at
   1-4 tokens on one card, flat in the token count. 48 layers x ~2.3 ms is ~110 ms per token, which was the PP=8
   decode time. Gathering only the routed experts' weights (`index_select` + `bmm`) costs 0.14 ms at 1 token, about
   0.1 ms per token, with the same error against an fp32 reference (`experiments-ubuntu2404/moe_gather_bench.py`).
   It is used while tokens x top_k <= 0.4 x experts (about 20 tokens); larger batches keep the fused op. Opt-in env
   `VLLM_HPU_MOE_GATHER_BF16=1` (patch 0005). The fused op itself failed (`synNodeCreateWithId ... moe_fwd_bf16`) at
   16 tokens when called without the plugin's chunk-size arguments.

Remaining gap: ~37 ms per token while device compute should now be a few ms per stage. Single-user runs vary between
14 and 27 tok/s; first-use compiles of new shapes are the likely cause (warmup is skipped because a full warmup is
~90+75 shapes x ~63 s, about 3 h, longer than the engine-ready timeout). Under PP the last stage spends most of its
time waiting for the sampled tokens and the other stages wait for the next step, i.e. the pipeline is latency-bound
at batch 1.

**Other findings on the way:**
- **Compiled PLE gives wrong outputs.** With the PLE layer compiled inside the decoder-layer graph, generated text turned
  repetitive and prompt logprobs were off by 4-8. The same code under `@torch._dynamo.disable` is correct. It stays
  disabled (it runs once per step, at layer 2 only).
- **Profiler:** Habana's light profiler wrapper strips HPU events from `key_averages()` (SW-224979) and breaks
  `export_chrome_trace` (wrong argument count). `hpu_worker.py` now saves `prof.profiler.kineto_results` directly,
  which keeps the device events (~850 MB per rank for 12 steps). Python stacks are off unless
  `VLLM_PROFILE_WITH_STACK=1`.
- **Startup:** weights load in ~2 min. The real checkpoint has only one PLE layer (`ple_layer_ids=[2]`). vLLM pads the
  hybrid KV layout with 11 layers ("may waste at most 29.7% KV cache").

Patches: `ports/qwen4_exp/vllm-gaudi-patches/0004-*` (PP, host PLE table, batched PLE prep, profiler) and `0005-*`
(gather MoE).

## 19. Next steps
- Qwen: find the remaining ~30 ms per token at batch 1 (device profile under PP), warm up the common decode shapes so
  benchmarks are not hit by compiles, then the full section 8 benchmark set.
- Hardware: run `hl_qual` NIC tests and open a support case for the port 9 flapping. With healthy links, TP=8 + gather
  MoE should be compared against PP=8.
- DeepSeek-V4.1-Flash Track A (section 15): CPU kernels validated; next `generate_hpu.py --selftest`, FP4->FP8
  conversion, then an 8-card run. It will use PP for the same reason.
