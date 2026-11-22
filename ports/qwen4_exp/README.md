# Qwen3.8-Flash-Next (Qwen4Exp) on Gaudi via vllm-gaudi

- `vllm-gaudi-patches/`: `git am` onto vllm-gaudi `2f1aaea0` (main, 2026-11-19). Adds `vllm_gaudi/models/qwen4_exp.py`
  and `qwen4_exp_mtp.py` (copies here for reading, synced at 0025) and registers the architecture, counts `indexed_attention` layers, and gives the PLE layer
  a KV-cache group in the HPU runner.
- `tests/`: tiny random-weight model builder and HF-vs-vLLM checks (prefill teacher-forced, decode), the HF-on-HPU
  reference run with its op workarounds (`hpu_compat.py`), and minimal repros of the Gaudi op gaps.
- Results: `../../results-ubuntu2404/qwen4_exp/VALIDATION.md`. Design and findings: FINDINGS.md section 14.
