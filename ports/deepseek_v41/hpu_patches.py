"""Patch DeepSeek-V4.1's reference `inference/model.py` to run on Gaudi (HPU) with kernel_torch.

Call `install()` BEFORE constructing `model.Transformer`, then build the model, call `move_hash_to_cpu(model)`, and
load weights with `load_checkpoint(model, path)` (instead of safetensors.torch.load_model).

Storage on the device (no float8/float4/e8m0 dtypes on the HPU):
* FP8 linears (attention, shared experts, ...): dequantized ONCE to bf16 at load; activations still go through the
  reference's FP8 fake-quant (act_quant) so numerics follow the reference.
* FP4 routed experts: raw packed bytes (uint8 [N, K/2]) + E8M0 scale bytes (uint8 [N, K/32]); decoded per call.
* Engram tables: raw FP8 bytes + E8M0 scale bytes; only the looked-up rows are decoded.
* Engram n-gram hashing: on the CPU (exact int64; Gaudi int64 arithmetic is inexact), ids sent to the device as int32.
"""
import os
import torch
import torch.nn.functional as F
from torch import nn

import kernel_torch as KT
import model as M

_U8 = torch.uint8


def _linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    assert bias is None
    qtype = getattr(weight, "qtype", None)
    if qtype == "fp4":
        xq, _ = KT.act_quant(x, M.fp8_block_size, M.scale_fmt)
        w = KT.dequant_fp4_weight(weight, weight.scale_u8, M.fp4_block_size)
        return F.linear(xq, w).to(torch.get_default_dtype())
    if qtype == "fp8":
        xq, _ = KT.act_quant(x, M.fp8_block_size, M.scale_fmt)
        return F.linear(xq, weight).to(torch.get_default_dtype())
    return F.linear(x, weight)


def _linear_init(self, in_features, out_features, bias=False, dtype=None):
    nn.Module.__init__(self)
    self.in_features, self.out_features = in_features, out_features
    dtype = dtype or M.default_dtype
    if dtype == torch.float4_e2m1fn_x2:
        self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=_U8), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(out_features, in_features // M.fp4_block_size, dtype=_U8),
                                  requires_grad=False)
        self.weight.qtype, self.weight.scale_u8 = "fp4", self.scale
    elif dtype == torch.float8_e4m3fn:
        # dequantized at load time (see load_checkpoint); the checkpoint's `.scale` is consumed there
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16), requires_grad=False)
        self.weight.qtype = "fp8"
        self.register_parameter("scale", None)
    else:
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
        self.register_parameter("scale", None)
    if bias:
        self.bias = nn.Parameter(torch.empty(out_features))
    else:
        self.register_parameter("bias", None)


def _engram_embed_init(self, num_embeddings: int, dim: int):
    nn.Module.__init__(self)
    self.num_embeddings, self.dim = num_embeddings, dim
    self.part_num_embeddings = (num_embeddings + M.world_size - 1) // M.world_size
    self.vocab_start_idx = M.rank * self.part_num_embeddings
    self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings
    self.block_size = M.fp8_block_size
    # ENGRAM_HOST=1 (default): the table stays in host RAM (47.5 GB per rank at mp=4 would not fit next to the experts);
    # only the looked-up rows are copied to the device and decoded there.
    self.host = os.getenv("ENGRAM_HOST", "1") == "1"
    dev = {"device": "cpu"} if self.host else {}
    self.weight = nn.Parameter(torch.empty(self.part_num_embeddings, dim, dtype=_U8, **dev), requires_grad=False)
    self.scale = nn.Parameter(torch.empty(self.part_num_embeddings, dim // self.block_size, dtype=_U8, **dev),
                              requires_grad=False)


def _engram_embed_forward(self, indices: torch.Tensor) -> torch.Tensor:
    indices = indices.int()
    mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
    local = torch.where(mask, torch.zeros_like(indices), indices - self.vocab_start_idx)
    flat = local.reshape(-1)
    if self.host:
        fc = flat.cpu().long()
        w_rows = self.weight.index_select(0, fc).to(indices.device, non_blocking=False)
        s_rows = self.scale.index_select(0, fc).to(indices.device, non_blocking=False)
    else:
        w_rows, s_rows = self.weight.index_select(0, flat), self.scale.index_select(0, flat)
    vals = KT.decode_e4m3(w_rows)  # bf16 [n, dim]
    scl = KT.decode_e8m0(s_rows).to(torch.bfloat16)  # [n, dim/32]
    out = (vals.unflatten(-1, (-1, self.block_size)) * scl.unsqueeze(-1)).flatten(-2)
    out = out.view(*indices.shape, self.dim).masked_fill(mask.unsqueeze(-1), 0)
    if M.world_size > 1:
        torch.distributed.all_reduce(out)
    return out


def install():
    M.act_quant = KT.act_quant
    M.fp4_act_quant = KT.fp4_act_quant
    M.sparse_attn = KT.sparse_attn
    M.hc_split_sinkhorn = KT.hc_split_sinkhorn
    M.linear = _linear
    M.Linear.__init__ = _linear_init
    M.ParallelEngramEmbedding.__init__ = _engram_embed_init
    M.ParallelEngramEmbedding.forward = _engram_embed_forward


def move_hash_to_cpu(model: nn.Module):
    """Run the engram n-gram hash state on the CPU and hand int32 ids back to the model's device."""
    hs = getattr(model, "engram_hash", None)
    if hs is None:
        return
    dev = next(p.device for p in model.parameters())
    hs.to("cpu")
    orig = hs.forward

    def forward(input_ids, start_pos, token_mask=None):
        tm = token_mask.cpu() if token_mask is not None else None
        ids = orig(input_ids.cpu(), start_pos, tm)
        assert int(ids.max()) < 2**31
        return ids.to(torch.int32).to(dev)

    hs.forward = forward


@torch.no_grad()
def load_checkpoint(model: nn.Module, path: str, rank: int = 0, mp: int = 1):
    """Load a converted shard into the patched model (byte views for quantized tensors, fp8 linears dequantized).

    `path` is either one `convert.py` file or a directory of `convert_stream.py` parts
    (`model{rank}-mp{mp}-partNNN.safetensors`); weights and their scales may sit in different parts."""
    import glob as _glob
    from contextlib import ExitStack
    from safetensors import safe_open
    files = ([path] if os.path.isfile(path) else
             sorted(_glob.glob(os.path.join(path, f"model{rank}-mp{mp}-part*.safetensors"))) or
             [os.path.join(path, f"model{rank}-mp{mp}.safetensors")])
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    seen = set()
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(fp, framework="pt", device="cpu")) for fp in files]
        where = {k: h for h in handles for k in h.keys()}
        for k, f in where.items():
            if k in seen:
                continue
            t = f.get_tensor(k)
            p = params.get(k, buffers.get(k))
            if k.endswith(".scale") and p is None:
                seen.add(k)  # consumed with its fp8 weight below
                continue
            if p is None:
                raise KeyError(f"checkpoint tensor {k} has no parameter in the model")
            if getattr(p, "qtype", None) == "fp8" and t.dtype == torch.float8_e4m3fn:
                s = where[k[: -len("weight")] + "scale"].get_tensor(k[: -len("weight")] + "scale")
                t = KT.dequant_fp8_weight(t.view(_U8), s.view(_U8), M.fp8_block_size)
            elif p.dtype == _U8:
                t = t.view(_U8)
            elif t.dtype != p.dtype:
                t = t.to(p.dtype)
            assert t.shape == p.shape, (k, t.shape, p.shape)
            p.data.copy_(t.to(p.device))
            seen.add(k)
    missing = [k for k in params if k not in seen]
    if missing:
        raise KeyError(f"{len(missing)} parameters not in checkpoint, e.g. {missing[:5]}")
