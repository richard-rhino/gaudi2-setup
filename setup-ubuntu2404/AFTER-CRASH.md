# Recovering from the habanalabs driver crash (FINDINGS section 16)

## 1. Reboot
`sudo reboot`. Nothing else clears the stuck CPU (soft lockup) and the D-state process. Model downloads resume where they
stopped, but both checkpoints finished overnight (Qwen3.8-Flash-Next 131/131 shards, DeepSeek-V4.1-Flash-1017 48/48, 06:05 UTC).

## 2. Check the internal links (port 9 was DOWN on 5 of 8 cards and flapping)
```
for b in 19 1a 43 44 b3 b4 cc cd; do echo -n "$b: "; hl-smi -i 0000:$b:00.0 -n link | grep DOWN | tr '\n' ' '; echo; done
sudo dmesg | grep -E "Port [0-9]+: link (down|up)" | tail
```
Healthy: only ports 8, 22, 23 DOWN (the external scale-out ports, no cables). If port 9 is still DOWN or flapping:
- Re-flashing the card firmware: see [FIRMWARE-UPDATE.md](FIRMWARE-UPDATE.md) (done 2026-11-20; firmware
  `fw-62.6.2` is already what driver 1.24.2 ships, so it is a re-flash, not an upgrade).
- Intel's qualification tool can test the NIC links: `/opt/habanalabs/qual/gaudi2/bin/hl_qual -h` (NIC/HCCL tests).
- Do NOT mask port 9 (`nic_ports_mask=0xFFFDFF`): tested 2026-11-20, HCCL hangs without it (FIRMWARE-UPDATE.md).

## 3. Avoid the trigger
Run vLLM in the default mode (`PT_HPU_LAZY_MODE=0`, eager + torch.compile). The crash happened while starting a
lazy-mode (`PT_HPU_LAZY_MODE=1`) HCCL context during link flaps. Always stop vLLM with the server's own shutdown or
`setup/stop-vllm.sh`; an orphaned worker pins the card memory.

## 4. Resume the model work
1. Qwen3.8-Flash-Next at TP=8: `bash experiments-ubuntu2404/qwen38fn-serve.sh` then
   `python3 experiments-ubuntu2404/compare_vs_nvidia.py` (greedy + logprob agreement against the deb-ai server).
2. Benchmarks with the section 8 scenarios.
3. DeepSeek Track A (`ports/deepseek_v41/`): run `test_kernels_cpu.py`, convert the checkpoint (`inference/convert.py
   --mp 8`), run the reference on 8 HPUs with `kernel_torch.py`.
