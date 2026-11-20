# DeepSeek-V4.1-Flash on Gaudi: preparation (FINDINGS.md section 15)

- `kernel_torch.py`: device-agnostic replacements for the 6 TileLang kernels of DeepSeek's `inference/kernel.py`
  (exact fp32 FP8/FP4 grid rounding, LUT byte decoders, chunked sparse attention with sink, Sinkhorn). Not yet validated
  (the driver crash blocked all torch processes); `test_kernels_cpu.py` is the validation suite.
- `kernel_helper.py`: the first draft written by a local DeepSeek-V4.1 (via `setup-ubuntu2404/ask-llm`), kept for
  comparison. Review notes are in FINDINGS 15 (FP4 tie rounding bug).
- `scale_spread.py`: measures FP4 expert scale spread on the checkpoint (numpy only). Result: max spread 5 => FP4->FP8
  per-channel conversion is bit-exact on Gaudi2.
- `repro_fp8cast.py`: checks whether HPU float8 casts match CPU (did not complete: the crash).
