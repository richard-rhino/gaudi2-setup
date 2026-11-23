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
    if os.getenv("ENGRAM_DEBUG") == "1" and M.rank == 0 and not getattr(self, "_dbg_done", False):
        self._dbg_done = True
        f = lambda t: (tuple(t.shape), str(t.dtype), str(t.device), float(t.float().abs().max()), bool(torch.isfinite(t.float()).all()))
        print("ENGRAM_DEBUG idx", indices.min().item(), indices.max().item(), "range", self.vocab_start_idx,
              self.vocab_end_idx, "rows", self.weight.shape, self.weight.device, "block", self.block_size, flush=True)
        for nm, t in (("w_rows", w_rows), ("s_rows", s_rows), ("vals", vals), ("scl", scl), ("out", out)):
            print("ENGRAM_DEBUG", nm, f(t), flush=True)
        print("ENGRAM_DEBUG s_rows bytes", s_rows.flatten()[:16].tolist(), "w bytes", w_rows.flatten()[:8].tolist(), flush=True)
    if M.world_size > 1:
        torch.distributed.all_reduce(out)
    return out


def _precompute_freqs_real(*args, **kwargs):
    """Gaudi has no complex dtype: the same frequencies as (cos, sin) pairs, [seq, rope_dim/2, 2] fp32."""
    dev = torch.get_default_device()
    with torch.device("cpu"):
        f = _orig_precompute(*args, **kwargs)
    return torch.view_as_real(f).contiguous().to(dev)


def _apply_rotary_emb_real(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Real-arithmetic version of model.apply_rotary_emb (in place, adjacent pairs as (re, im))."""
    y = x
    xf = x.float().unflatten(-1, (-1, 2))
    a, b = xf[..., 0], xf[..., 1]
    c, s = freqs[..., 0], freqs[..., 1]
    if inverse:
        s = -s
    shape = (1, x.size(1), -1) if x.ndim == 3 else (1, x.size(1), 1, -1)
    c, s = c.reshape(shape), s.reshape(shape)
    y.copy_(torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2))
    return y


def _on_default_device(fn):
    cache = {}

    def wrapper(*args, **kwargs):
        dev = torch.get_default_device()
        key = (args, tuple(sorted(kwargs.items())), str(dev))
        t = cache.get(key)
        if t is None:  # a fresh H2D copy per call cost a host sync per layer
            if len(cache) > 64:
                cache.clear()
            t = cache[key] = fn(*args, **kwargs).to(dev)
        return t
    wrapper._on_default_device = True
    return wrapper


_orig_precompute = None


def install():
    global _orig_precompute
    if _orig_precompute is None:
        _orig_precompute = M.precompute_freqs_cis
    M.precompute_freqs_cis = _precompute_freqs_real
    for fn in ("get_window_topk_idxs", "get_dspark_topk_idxs"):  # lru_cached; cache ignores the device
        if hasattr(M, fn) and not getattr(getattr(M, fn), "_on_default_device", False):
            setattr(M, fn, _on_default_device(getattr(M, fn)))
    M.apply_rotary_emb = _apply_rotary_emb_real
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


# --------------------------------------------------------------------------- MoE: stacked experts, gather decode
MOE_GATHER_MAX = int(os.getenv("DSV41_MOE_GATHER_MAX", "32"))  # T*topk up to this -> gather path


@torch.no_grad()
def stack_experts(model: nn.Module):
    """After loading: stack each MoE's local FP4 experts into [E_local, ...] byte tensors (w1/w3/w2 + scales) and
    free the per-expert copies, so the forward can gather routed experts without a data-dependent loop."""
    for moe in [m for m in model.modules() if isinstance(m, M.MoE)]:
        local = [moe.experts[i] for i in range(moe.experts_start_idx, moe.experts_end_idx)]
        if not local or getattr(local[0].w1.weight, "qtype", None) != "fp4":
            continue
        for nm in ("w1", "w3", "w2"):
            w = torch.stack([getattr(e, nm).weight.data for e in local])
            sc = torch.stack([getattr(e, nm).scale.data for e in local])
            for e in local:  # free the originals right away (peak = one extra matrix stack)
                lin = getattr(e, nm)
                lin.weight.data = torch.empty(0, dtype=_U8, device=w.device)
                lin.scale.data = torch.empty(0, dtype=_U8, device=w.device)
            setattr(moe, f"_{nm}", w)
            setattr(moe, f"_{nm}_s", sc)
        moe._stacked = True


def _experts_apply(moe, x, w_idx, wts):
    """Routed-expert FFN for row-aligned x [N, dim], expert ids w_idx [N] (local), weights wts [N, 1] (fp32).
    Same numerics as Expert.forward + _linear: fp8 act fake-quant, FP4 weights decoded exactly, fp32 SwiGLU."""
    blk4 = M.fp4_block_size
    xq, _ = KT.act_quant(x, M.fp8_block_size, M.scale_fmt)
    w1 = KT.dequant_fp4_weight(moe._w1.index_select(0, w_idx), moe._w1_s.index_select(0, w_idx), blk4)
    w3 = KT.dequant_fp4_weight(moe._w3.index_select(0, w_idx), moe._w3_s.index_select(0, w_idx), blk4)
    gate = torch.bmm(w1, xq.unsqueeze(-1)).squeeze(-1).to(torch.get_default_dtype()).float()
    up = torch.bmm(w3, xq.unsqueeze(-1)).squeeze(-1).to(torch.get_default_dtype()).float()
    lim = moe.experts[moe.experts_start_idx].swiglu_limit
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = (wts * (F.silu(gate) * up)).to(x.dtype)
    hq, _ = KT.act_quant(h, M.fp8_block_size, M.scale_fmt)
    w2 = KT.dequant_fp4_weight(moe._w2.index_select(0, w_idx), moe._w2_s.index_select(0, w_idx), blk4)
    return torch.bmm(w2, hq.unsqueeze(-1)).squeeze(-1).to(torch.get_default_dtype())


def _moe_forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
    if not getattr(self, "_stacked", False):
        return _orig_moe_forward(self, x, image_mask)
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
    T, K = indices.shape
    E = self.experts_end_idx - self.experts_start_idx
    local = indices - self.experts_start_idx
    valid = (local >= 0) & (local < E)
    y = torch.zeros_like(x, dtype=torch.float32)
    if T * K <= MOE_GATHER_MAX:  # decode: every (token, k) pair at once, no host sync
        y = _moe_gather_fn()(self, x, weights, local, valid)
    else:  # prefill: loop over the experts that got tokens (as the reference), on the stacked weights
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            li = torch.full((idx.numel(),), i - self.experts_start_idx, dtype=torch.long, device=x.device)
            y[idx] += _experts_apply(self, x[idx], li, weights[idx, top, None].float()).float()
    if M.world_size > 1:
        torch.distributed.all_reduce(y)
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


def _moe_gather(self, x, weights, local, valid):
    T, K = local.shape
    li = torch.where(valid, local, torch.zeros_like(local)).reshape(-1)
    wts = (weights.float() * valid).reshape(-1, 1)
    out = _experts_apply(self, x.repeat_interleave(K, 0), li, wts)  # [T*K, dim]
    return (out.float() * valid.reshape(-1, 1)).view(T, K, -1).sum(1)


def _moe_gather_fn():
    return _COMPILED.get("moe_gather", _moe_gather)


_COMPILED: dict = {}
_orig_moe_forward = None


_OFFSET_SEEN = set()


def _offset_safe(fn, name):
    """A compiled HPU graph keeps the storage offset of each input it was recorded with, and a later call with a view
    at another offset fails ("offset_ 0 is not matching with the offset of new tensor"). Give it offset-0 copies;
    report each function the first time, with the size, so the cost of the copies is visible."""
    def run(*args, **kwargs):
        out = []
        for i, t in enumerate(args):
            if isinstance(t, torch.Tensor) and t.device.type == "hpu" and t.storage_offset() != 0:
                if (name, i) not in _OFFSET_SEEN:
                    _OFFSET_SEEN.add((name, i))
                    print(f"[offset_safe] {name} arg {i}: view at offset {t.storage_offset()}, shape {tuple(t.shape)}, "
                          f"{t.numel() * t.element_size() / 2**20:.2f} MiB copied", flush=True)
                t = t.clone()
            out.append(t)
        return fn(*out, **kwargs)
    return run


def install_compile():
    """DSV41_COMPILE=1: torch.compile (hpu_backend, static shapes) the pieces whose shapes do not depend on the
    position: hyper-connection mixes/Sinkhorn, RMSNorm, the quantized linear, sparse attention, the MoE gather."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 8192
    c = lambda f: _offset_safe(torch.compile(f, backend="hpu_backend", dynamic=False), f.__qualname__)
    M.Block.hc_mixes = c(M.Block.hc_mixes)
    M.Block.hc_pre = c(M.Block.hc_pre)
    M.Block.hc_post = c(M.Block.hc_post)
    M.RMSNorm.forward = c(M.RMSNorm.forward)
    M.linear = c(_linear)
    M.sparse_attn = c(KT.sparse_attn)
    _COMPILED["moe_gather"] = c(_moe_gather)
    # Opt-in: ~8% faster decode, but 4 vs 2 of 132 top-1 differences from rack-n03 (FINDINGS 20); too few to rank.
    if os.getenv("DSV41_ATTN_STATIC", "0") == "1":
        import attn_static  # decode-time attention with bucketed shapes (checked by test_attn_static.py)
        attn_static.install(compile_pieces=True)


def install_moe():
    global _orig_moe_forward
    if _orig_moe_forward is None:
        _orig_moe_forward = M.MoE.forward
        M.MoE.forward = _moe_forward
