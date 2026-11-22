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
- **Update 2026-11-22:** likely cause found: PHY resets of the unused external port 8 (brought up by netplan DHCP) make
  the driver re-init port 9. With the d8 netdevs down the flapping stopped in a first short test. A persistent networkd rule is installed
  (`setup-ubuntu2404/05-gaudi-external.network`), not yet checked across a reboot.
- Full write-up with per-card counts, FEC counters and what was tried: [setup-ubuntu2404/NIC-LINK-FLAPPING.md](setup-ubuntu2404/NIC-LINK-FLAPPING.md).
  Summary: only port 9 goes down (~299k events in 28 h, 72% on card 0 / 43:00.0). It looks like port 9 is the
  card k <-> k+4 link. 33 internal ports have >1,000 uncorrectable FEC codewords. BIOS 2.3 / BMC 01.07.05 / card fw
  62.6.2 are all the latest.
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
  repetitive and prompt logprobs were off by 4-8. The same code under `@torch._dynamo.disable` is correct. It was
  disabled at this point. Section 19 found the compiler bug and a rewrite, so it is compiled again now.
- **Profiler:** Habana's light profiler wrapper strips HPU events from `key_averages()` (SW-224979) and breaks
  `export_chrome_trace` (wrong argument count). `hpu_worker.py` now saves `prof.profiler.kineto_results` directly,
  which keeps the device events (~850 MB per rank for 12 steps). Python stacks are off unless
  `VLLM_PROFILE_WITH_STACK=1`.
- **Startup:** weights load in ~2 min. The real checkpoint has only one PLE layer (`ple_layer_ids=[2]`). vLLM pads the
  hybrid KV layout with 11 layers ("may waste at most 29.7% KV cache").

Patches: `ports/qwen4_exp/vllm-gaudi-patches/0004-*` (PP, host PLE table, batched PLE prep, profiler) and `0005-*`
(gather MoE).

## 19. PP=8 decode: from ~27 to ~32 tok/s, and steady (2026-11-21)
**Where the time went.** `VLLM_PP_STAGE_TIMING=1` (vllm-gaudi patch) records HPU events around each stage's recv and
forward and logs means every 100 passes. Nothing is synchronized, so the timing does not change the schedule. A
first version that called `torch.hpu.synchronize()` around the recv hung the first request for 5 minutes. Batch-1
decode before the fixes:

| PP stage | 0 | 1-6 | 7 |
|---|---|---|---|
| device ms per token | **~16** | ~2.6 each | ~5 (incl. hyper-connection mixer) |

The other ranks are idle most of the time (py-spy: waiting on the shm RPC queue). The last rank spends 91% in the
`.cpu()` of the sampled token, so the pipeline is a serial chain of device time.

**Stage 0 = the one PLE layer.** The model has a single PLE layer (`ple_layer_ids=[2]`), and stage 0 holds it.
Per-layer events (`QWEN4_LAYER_TIMING=1`): the PLE layer took ~10.4 ms against ~0.9 ms for a normal layer, plus
~2.8 ms of host prep (1.6 ms of that is the `.cpu()` of the conv state, 0.5 ms hashing). The PLE forward ran eager
(`@torch._dynamo.disable`), i.e. ~40 separately launched small ops, because compiling it inside the decoder-layer
graph gave wrong outputs (section 18).

**The miscompile, narrowed down** (`ports/qwen4_exp/ple_compile_repro.py`): `sigmoid(gate) * value.unsqueeze(-2)`
is wrong at T=128 (max error 3.7 on values of size 3.4) when `gate` comes from the in-graph
`abs().clamp_min().sqrt() * sign()` chain. Each op alone is fine, `gate` passed in from outside is fine, and decode
shapes (T=1) are fine. That is why decode looked right but prompt logprobs were off by 4-9. Writing the same product
as `einsum("tk,th->tkh", ...)`, `bmm` or a `cat` of 4 products gives correct results. The PLE math is now its own
compiled graph (`_ple_math`, einsum form). State loads and stores (bitcast views, `index_copy_`) stay eager.
`QWEN4_PLE_COMPILE=0` restores the old path. PLE layer: 10.4 -> 3.5 ms. Correctness vs deb-ai is unchanged
(`results-ubuntu2404/qwen38fn/compare_vs_deb-ai-pp8-gather-plecompile.txt`: prompt |dlogprob| mean 0.02-0.15; greedy
output is identical for as long as in the earlier runs).

**The 16-vs-29 tok/s runs were the port-9 links.** Workers open whichever card is free first, so the PP rank -> card
order is random on each start. Timing showed the slow windows always add ~28-44 ms per token on the hops between card
pairs k <-> k+4, which port 9 connects (details: `setup-ubuntu2404/NIC-LINK-FLAPPING.md`).
`VLLM_HPU_MODULE_ORDER=0,1,...,7` (vllm-gaudi patch, set by `qwen38fn-serve.sh`) spawns worker i with
`HLS_MODULE_ID=i`. `HABANA_VISIBLE_MODULES` does not work for this; see the NIC page. With that order no hop uses a
port-9 pair: 11 of 11 warm single-user runs gave **31.7-32.3 tok/s**
(`results-ubuntu2404/qwen38fn/single_user_pp8_pinned.txt`).

Batch-1 device time per token now: stage 0 ~9 ms, stages 1-6 ~2.6 ms, stage 7 ~4.6 ms, ~0.1 ms per hop, ~31 ms in
total. What is left on stage 0: the PLE layer at 3.5 ms (vs ~0.9 for a normal layer) and the host prep at ~2.4 ms.
The prep could drop most of its 1.6 ms sync by keeping the last 2 context tokens per request on the host.

**Section 8 benchmark set, PP=8 pinned** (`experiments-ubuntu2404/office-bench.sh`, 2026-11-21). The same
server ran the set twice. The first ("cold") pass is mostly compile time: each new prompt-length or batch bucket
compiles for 1-4 minutes because `VLLM_SKIP_WARMUP=true`
(`results-ubuntu2404/office-qwen38fn-pp8-pinned-cold.md`). Second pass, everything compiled
(`results-ubuntu2404/office-qwen38fn-pp8-pinned-warm.md`):

| Scenario | Users | Prompt / answer tok | Per-user tok/s p50 (p99 user) | TTFT p50 / p99 (s) | Aggregate out tok/s | Total tok/s |
|---|---|---|---|---|---|---|
| single | 1 | 128 / 256 | 30.9 (30.7) | 0.3 / 0.3 | 30.0 | 47 |
| chat-10 | 10 | 1024 / 512 | 11.9 (8.9) | 0.6 / 1.8 | 105.7 | 325 |
| chat-16 | 16 | 1024 / 512 | 10.7 (3.6) | 0.6 / 79.2 | 107.6 | 325 |
| agentic-10-24k | 10 | 24576 / 1024 | 11.2 (9.4) | 3.7 / 19.2 | 100.2 | 2449 |
| mixed-16 | 16 | 8192 / 768 | 9.3 (8.4) | 1.4 / 10.4 | 139.6 | 1648 |

chat-16's p99 TTFT (79 s) is one more shape that still compiled in the second pass. With 10+ users the aggregate
levels off at ~100-140 tok/s. Per decode step PP runs one micro-batch through 8 serial stages, so batching helps
probably only until the gather MoE path (<= ~20 tokens) gives way to the fused op that reads all experts (not measured yet).

## 20. DeepSeek-V4.1-Flash on the real checkpoint (2026-11-21/22)

DeepSeek's reference `inference/model.py` runs on 4 cards (modules 0-3, model parallel 4) through
`ports/deepseek_v41/`: streaming converter, torch kernels, engram tables in host RAM, real-valued RoPE.

**What the output check does and does not show.** Three greedy prompts, 48 tokens, eager, against rack-n03
(`results-ubuntu2404/dsv41-mp4-first-correct.txt`, `dsv41-reference-rack-n03.txt`):

| Prompt | Gaudi vs rack-n03 |
|---|---|
| Capital of Japan | identical, including the stop |
| Fibonacci function | differs from the first line (another title and intro); same section, signature and docstring after that |
| Why the sky is blue | same start, then "(blue light)" vs "(blue)" and a different second sentence |

So the port gives coherent output close to the reference, but **agreement with the reference is not established**.
The commit message of 3328934 ("runs correctly") and the file name `*first-correct.txt` claim more than was measured.

Speed of that first eager run: 0.14 tok/s.

### TP=8 re-test after the port 8 fix (section 17), 2026-11-22
Qwen3.8-Flash-Next, TP=8 PP=1, gather MoE, single user, 256 tokens: **27.7-29.7 tok/s on 9 of 9 warm runs** (median
28.1), TTFT 0.22-0.31 s, 0 port 9 link-downs. Raw: `results-ubuntu2404/qwen38fn/single_user_tp8_port8fix_gather.txt`.
The old 2.6-3.4 tok/s at TP=8 was measured with the fused MoE and the uncompiled PLE layer as well as the flapping
links, so not all of the ~9x is the link fix. PP=8 still does ~32 tok/s single user.

**Where the ~9x comes from** (single user, warm, 256 tokens, median):

| TP=8 configuration | tok/s | ms per token | Step |
|---|---|---|---|
| Before: flapping links, fused MoE, eager PLE | 2.6-3.4 | 294-385 | |
| Port 8 fix, fused MoE (`single_user_tp8_port8fix_fused.txt`) | **5.7** (5.5-6.0, 9 runs) | 175 | **1.7-2.2x**: the link fix, plus a little from the compiled PLE (~7 ms per token) |
| Port 8 fix, gather MoE | **28.1** (27.7-29.7, 9 runs; other launches: medians 24.5-29.2) | 36 | **4.3-5.1x**: the gather MoE alone |

So the larger factor is the gather MoE, and the link fix is what makes TP=8 stable (no stalls, no hangs).

**Soak, 16 users, 21 min** (random 1024 in / 256 out, 10 passes of 96 requests after a warm pass,
`bench_tp8_port8fix_16users.txt` in the same directory):

| | TP=8, port 8 fix | PP=8, port 8 fix (re-run afterwards, `*pp8_port8fix*`) |
|---|---|---|
| Aggregate output | **214.8 tok/s** (213.5-218.8 over 10 passes) | **197.7 tok/s** (187.7-204.7 over 3 passes) |
| Single user, median | 24.5-29.2 tok/s by launch (28.1 in the soak's launch) | **31.6 tok/s** (31.0-31.9, 9 runs, 1 launch) |
| TPOT p50 | 66-68 ms | 72-83 ms |
| TTFT p50 / p99 | ~2.0 s / 5.3-5.7 s | **1.3-1.6 s / 2.5-2.6 s** |
| ITL p50 / p99 | 54-55 ms / 627-671 ms | 68-78 ms / **84-91 ms** |

The 123 tok/s in section 18 is obsolete for PP=8. At 16 users the two can't be ranked: TP measured 9% higher, but that
is one launch each, and TP=8 alone varies by up to ~19% from launch to launch (medians 24.5-29.2). PP is ahead
for one user (8-29%, against the fastest to slowest TP launch) and has far better tail latency. **PP=8 stays the default for interactive serving**; TP=8 is the simpler
base for MTP. The PP soak was short (3 passes, the last one the slowest), so it is not a stability test.

- 0 of 960 requests failed, 0 port 9 link-downs, no server errors.
- The 10 s server-log windows under 100 tok/s all fall at the start of a pass (idle client, then 16 prefills at once) or
  in the single-user test afterwards. None is mid-pass, so there was no sign of the old 4-16 s all-reduce stalls.
- Single user afterwards: 28.0-28.7 tok/s, no degradation.
- Against deb-ai, prompt-token |dlogprob| means are 0.02-0.19, in line with PP=8
  (`compare_vs_deb-ai-tp8-port8fix.txt`).
- The cold warm-up pass was 29 tok/s aggregate with a 56 s median TTFT: first-use compiles are still the weak spot.
- All without MTP.

## 21. Next steps

To-do, in working order:
- [x] **TP=8 soak** (16 users, 21 min) after the port 8 fix.
- [x] TP=8 single-user test with the gather MoE off: 5.7 tok/s, so the gather MoE is 4.9x and the link fix ~2x.
- [ ] **Qwen3.8-Flash-Next MTP** (requested 2026-11-22, ahead of DeepSeek). All Gaudi numbers so far are without it; deb-ai gets 2.4-2.6
      tokens per step with it. Scope, from reading the code (nothing run yet):
  - The checkpoint has the head (31 `mtp.*` tensors); `qwen4_exp.py` drops them today (`"mtp.": None`).
  - Port upstream `vllm/models/qwen4_exp/nvidia/mtp.py` (508 lines) to the HPU layers and register `Qwen4ExpMTP`.
    It takes the hyper-connection streams (hc_count x hidden) before the final mixer and returns two hidden streams.
  - The plugin has a generic path to build on: `hpu_model_runner` sends method `mtp` to `HpuEagleProposer`. Speculative
    decoding is not supported with async scheduling, and not fully with torch.compile.
  - **Main blocker: recurrent state cannot be rewound on HPU.** The plugin's GDN kernels (36 of the 48 layers) raise
    NotImplementedError for speculative decoding. A KV cache rewinds by position; GDN and PLE state do not.
    Design: 1 + num_spec state slots per request, save after every token, resume from the last kept token's slot.
    - [x] GDN conv + SSM ops for this (`hpu_gdn_spec_decode.py`, patch 0007): 24/24 CPU cases, random rejections.
    - [x] The same for the PLE conv history and n-gram context tokens (patch 0008): 24/24 CPU cases.
    - [x] Runner and GDN layer: 1 + num_spec compact slots per request, load = last kept, store = one per token
          (patch 0009; slot arithmetic CPU-tested, speculative path not run on HPU).
    - [x] Slowdown check for patches 0007-0009 without MTP: **none detectable** (`single_user_tp8_mtp_slots_ab.txt`).
          Old/new/old/new fresh launches, 9 warm runs each: means 26.2 / 27.1 / 29.0 / 26.6 tok/s. New vs old is -2.6%,
          old vs old 10.7%, so a difference of a few percent can't be ruled out. The generated text was identical.
          Lesson: compare TP=8 speeds over several launches; runs within one launch understate the variation.
    - [x] Runner: positions and slots of padded multi-token decode steps (patch 0010, CPU-tested, not run on HPU).
          Two bugs in the existing plugin path, both only hit when a step decodes >1 token per request:
          padded tokens of shorter requests overwrote that request's position-0 KV slot, and mrope positions
          were replaced by plain ones. Single-token steps are identical to before (2000 random steps).
    - [x] Target model: return the pre-mixer streams, handed over through the aux hidden state path (patch 0011;
          model forward CPU-tested with stub layers, not run on HPU).
    - [x] Proposer: each draft step gets the previous step's multi streams, and 3-axis positions work, text only
          (patch 0012; CPU-tested with a stub draft model, not run on HPU).
    - [x] Draft model written: one attention layer (dense on HPU, like the target), no GDN, no PLE (patch 0013;
          weight routing and input fusion CPU-tested, never loaded).
    - [ ] First run on HPU: load, then compare greedy output and logprobs with the non-MTP run.
          Launch 1 (TP=8, k=3): the draft model loads (0.96 GB per card), then the first request failed on a
          too-narrow state block table. Fixed in patch 0014, with the proposer's choice of block table.
          Launch 2: the graph compiler rejected the GDN rewind ops inside the decoder layer ("missing a control
          edge": several in-place updates of one state tensor per graph). A standalone compiled test did not
          reproduce it. Patch 0015 writes each state once.
          Launch 3: failed in the first attention layer of the drafted decode step, which the plugin took for a
          prompt of 4 tokens (it reads the batch size from a per-request length tensor). Fixed in patch 0016.
          Launch 4: first text with MTP, 0015 holds in the full model. One prompt leaves deb-ai's text after 28
          tokens. First reported here as an MTP bug, which was premature: deb-ai is the wrong reference (Gaudi
          without MTP also differs from it). Launch 5: same token for every max_tokens, and the 4 compare prompts
          match non-MTP Gaudi for the 120 saved chars each. A token-level test against non-MTP Gaudi decides it.
          Requests with logprobs crash in the rejection sampler (token ids on the CPU).
          Launch 6, token-level test (`mtp_equivalence_*.json`, 16 prompts x 128 tokens, TP=8): the suspected prompt was
          fine (non-MTP Gaudi gives the same text, margin 0.25). But MTP had a **real bug**: 5/16 identical, and 2
          flips at margins 7.6 and 7.8, at tokens computed from positions 896 and 129. A drafted token that opened a new
          128-token KV block got the first token's block list (a plugin bug, patch 0017, CPU repro on the runner's
          lines: 2270 wrong rows before, 0 after). Noise floor: non-MTP sequential vs 4 at a time is 6/16 identical,
          with flips at margins up to 0.375.
          **Speed with that code: MTP is slower**, 15.7 vs 24.5 tok/s single user (medians, 1 launch each) at 2.56-2.61
          tokens per step, so a step costs ~4x a non-MTP step. TTFT ~0.8 vs ~0.27 s. Not yet profiled.
          Launch 7, with 0017 (same saved non-MTP tokens as reference): 8/16 identical one at a time, 5/16 with 4
          users at a time. Both large-margin flips are gone: the first differences are at margins up to 0.4375 (one
          at a time) and 0.25 (4 users), against up to 0.375 between the two non-MTP runs, so one is a step past that
          noise sample. **Limits of this test:** tokens are comparable only up to a prompt's first difference, which
          is 1330 of 1812 and 953 of 1868 tokens, with 7 and 4 block crossings among them. Not tested: images,
          logprobs (crash), more than 4 users, contexts over ~1k tokens, PP=8, warm-up.
          Speed is unchanged: **15.3 tok/s** (14.9-15.9) vs 24.5 (24.2-25.6) without MTP, 1 launch each, single user;
          ~171 ms per step against ~41, where break-even at 2.61 tokens per step is ~107 ms.
          Profile (`mtp_step_profile_tp8.txt`): drafting took 129 of 209 ms, because the plugin compiled only the target
          model and ran the drafter op by op. **Patch 0018 compiles the drafter:** drafting 42 ms, step 108 ms (with timers).
          Launch 8, with 0018: 9/16 and 8/16 identical (1 and 4 users), largest flips 0.4375 and 0.25, i.e. no worse;
          draft acceptance 61.4% (62.2% before). **34.4 tok/s** single user (31.7-38.3), TTFT 0.25 s. That is one launch
          against non-MTP launches of 24.5-29.2, so the size of the gain needs repeated launches. With 4 users the test
          finished at about the same rate as without MTP (38 vs 39 tok/s, compiles included), so a multi-user gain is open.
          A/B with 0018 (`mtp_ab_tp8.txt`; TP=8, six launches, order 0 3 3 0 0 3, warm passes only):
          - single user: launch medians 25.8/26.1/26.4 -> 31.8/32.2/32.6 tok/s, **+23%** (+20.5 to +26.4% over the pairings),
            TTFT 0.26 -> 0.28 s. That is the level of PP=8 without MTP (31.6, one launch).
          - 16 users, real text: 176/182/191 -> 129/156/162 tok/s, **-14%** (-8 to -32%).
          - 16 users, random tokens 1024/256 (one launch each): 204.5 -> 123.7 tok/s; TTFT p50 2.0 -> 0.72 s, p99 5.4 -> 7.0-7.5 s.
          - **Output at 16 users is NOT verified, and something is off:** from the same 48 requests MTP wrote 9718 tokens
            per pass against 9206 (12 passes each, all three MTP launches above all three others), at 74% draft acceptance
            (53% on the single-user prompts). The equality test only went to 4 users and 128 tokens.
          - **Tested afterwards at 16 users and 256 tokens** (`mtp_equivalence256_*.json`): no sign of an MTP bug.

            | Pair | Identical | Tokens before first diff | Largest margin at a flip | Output tokens |
            |---|---|---|---|---|
            | no MTP vs no MTP, 16 users twice | 7/16 | 1500 | 0.25 | 3385, 2790 |
            | no MTP: gather vs fused MoE, 16 users | 5/16 | 1353 | 0.75 | 3385, 3273 |
            | MTP vs no MTP, 1 user | 9/16 | 1955 | 0.4375 | 2890, 3097 |
            | MTP vs no MTP, 16 users, run a | 6/16 | 1757 | 0.25 | 3317, 3385 |
            | MTP vs no MTP, 16 users, run b | 5/16 | 1281 | 0.5 | 3296, 2790 |

            Answer lengths hang on near ties (one prompt ends after 6 tokens or runs to 256, with or without MTP), so
            ~512 extra tokens is 2-3 of the 48 requests taking the long branch; MTP's totals here are inside the range
            without it. No extra repetition (the one new looping prompt also loops on fused MoE without MTP). The high
            acceptance is the prompts: launch 8 already accepted 70.8% on them with 1 and 4 users. Limit: after a
            prompt's first difference nothing is compared.
    - [ ] **MTP is not servable yet (JSON schema crashes it). The plugin's HPU rejection sampler (`v1/sample/hpu_rejection_sampler.py`).** Found by
          reading it, reproduced on CPU with the real code, fixed by 0019:
          1. `assert sampling_metadata.all_greedy`: any request with temperature > 0 raises in the workers. This model's
             default is temperature 1.0, so that includes a plain chat request. **Every MTP result above is temperature 0.**
          2. Logprobs: token ids are left on the CPU (the crash of launch 4), the runner drops the spec-decode logprobs,
             and in a padded batch they would be read from other requests' rows (the vendored code expects compact rows).
          3. In a padded batch a request whose drafts were all accepted loses its bonus token (the padding counts as a
             mismatch): right tokens, fewer per step. The 16-user speeds above include that loss.
          **Fix 0019 (exported)**, on
          the cards at TP=8, one launch each (`mtp_sampling_tp8.txt`):
          - works: chat with defaults, temperature, top_p, logprobs (greedy, sampled, chat), penalties, seeds, 16 mixed
            requests at once. Logprobs agree with the non-MTP run as closely as that run agrees with itself (top-2 margin
            difference 0.20 vs 0.21 mean), prompt logprobs vs deb-ai as without MTP. Greedy: 9/16 and 3/16 identical
            (1 and 16 users), largest flips 0.4375 and 0.75, the latter what switching the MoE path does without MTP.
          - **sampled requests are slower with MTP: 22.0 tok/s (20.0-23.2) vs 25.0 (24.5-25.7) without.** The +23% is
            greedy only. Greedy was 29.4 tok/s in this launch (31.8-32.6 in the A/B before 0019), cause not known yet.
          - the sampled distribution, open at first (960 completions, p = 0.03-0.04 for tokens 2-4), was the stock
            sampler's bias, see the next item. Correct since 0020.
          - **still crashes: structured output (JSON schema)**, `UnboundLocalError: sampling_metadata`. The plugin's
            structured-output path does not handle speculative decoding.
    - [ ] **All sampled output on this stack is biased, with or without MTP: `Tensor.exponential_` on HPU.** Found while
          testing 0019 against exact probabilities (`results-ubuntu2404/hpu-exponential-sampling-bias.txt`, one card).
          The values have the right marginal distribution but are not independent within a row, and vLLM samples with
          `argmax(probs / exponential noise)`. In a test distribution the compiled stock sampler was off by TVD
          0.054-0.065 (about 5-6% of the probability on the wrong tokens; unbiased would be 0.008). CPU is exact, and so
          is HPU `torch.rand`. **Fix 0020** (exported): noise from `torch.rand`; then exact
          (p = 0.34, 0.94; all rejection-step positions p >= 0.57; seeded 0.69 over 72000 draws), and two cards still
          draw identical tokens from one seed, which TP needs. Greedy results are not affected. Not checked: lazy mode,
          other Gaudi software versions. Not reported upstream.
    - [x] **0019-0021 on the full server (TP=8): sampling is now correct in every check, with and without MTP**
          (`results-ubuntu2404/qwen38fn/mtp_sampling_dist2_tp8.txt`; 6400 completions per launch, each token against
          the probabilities of its own step; 4 non-trivial checks per launch).
          - before 0020: fails 2 of 4 (p = 0.000 and 0.002; first token TVD 0.029, unbiased would be 0.010).
          - with 0020+0021: p = 0.50-0.99. With MTP: p = 0.44-0.90.
          - A first run against a prefill-path reference had one failure with 0020 (p = 0.006,
            `mtp_sampling_dist_tp8.txt`). With each launch's own probabilities the same check gives p = 0.99, so that
            was the reference.
          - 15 draws of the three launches (5 of them without 0020) are one or two ranks outside their computed top-p
            set. The stock top-k keeps tokens tied with the 20th, and 1-2 ties would explain each one; the API returns
            20 logprobs, so the ties themselves were not seen.
          - **0020 as first written was a speed regression, fixed by 0021:** one seeded request made it draw every row
            per generator (26-28 ms instead of 5.5 ms per call; seeded single user 14.6 tok/s at TP=8,
            `results-ubuntu2404/hpu-seeded-sampling-speed.txt`). All requests seeded still costs ~27 ms, as before 0020.
          - Single user, median tok/s, one launch each (launches differ by up to ~19%):

            | Request | old sampler | 0020+0021 | 0020+0021, MTP k=3 |
            |---|---|---|---|
            | greedy | 27.9 | 25.45 | 29.3 |
            | sampled | 27.25 | 25.55 | 23.85 |
            | sampled, seeded | 24.45 | 23.0 | 22.55 |

            Greedy does not use the changed code and dropped as much as the sampled columns, so 0020+0021 cost nothing
            measurable here. **MTP is still no faster for sampled requests.** The seeded loop in 0019 was suspected for
            that and is not the reason (22.0 before 0021, 22.55 after).
          - `ruff` and `yapf` are not installed here, so 0019-0021 were not lint-checked.
    - [ ] Model card in tests/models for the draft model, once it runs.
  - Start at TP=8 (no pipeline to thread the hidden states through), then try PP=8.
  - Done when: greedy output and logprobs agree with the non-MTP run, with acceptance rate and tok/s reported for
    single and 16 users.
- [ ] **DeepSeek-V4.1-Flash** (section 20): run `compare_modes.py` with the static-shape attention (speed, and whether
      each speed-up changes the output), then compare logprobs with rack-n03.
- [ ] Qwen: warm up the common prefill shapes (a new prompt length compiles for 1-4 minutes); keep the PLE context
      tokens on the host.
- [ ] Check the port 8 fix across a reboot.
