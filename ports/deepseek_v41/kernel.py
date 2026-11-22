"""Stand-in for DeepSeek's TileLang `inference/kernel.py` (TileLang does not run on Gaudi).

`model.py` does `from kernel import ...`; this directory comes first on sys.path, so it gets these PyTorch versions.
The FP8/FP4 GEMMs are never called: hpu_patches replaces `model.linear` (weights are decoded to bf16 instead)."""
from kernel_torch import act_quant, fp4_act_quant, hc_split_sinkhorn, sparse_attn  # noqa: F401


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("fp8_gemm: hpu_patches.install() should have replaced model.linear")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("fp4_gemm: hpu_patches.install() should have replaced model.linear")
