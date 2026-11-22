"""Device-agnostic PyTorch replacement for DeepSeek-V4.1 inference/kernel.py (TileLang), for Gaudi (HPU).

Differences from a naive port, all deliberate:
* No float8/float4 dtypes ever live on the device (Gaudi2 FP8 is a different E4M3 variant, max 240, and the
  fp4/e8m0 dtypes are not supported). Quantization is emulated with exact fp32 grid rounding
  (round-half-to-even, like the hardware casts); quantized weights are kept as raw uint8 bytes and decoded
  with 256-entry lookup tables.
* GEMMs dequantize to bf16 (FP8/FP4 values times power-of-two scales are exact in bf16) and use bf16 matmul
  with fp32 accumulation, instead of fp32 matmul.
* sparse_attn processes queries in chunks to bound the [chunk, topk, d] gather.
Public functions keep the reference names/semantics; the GEMM/quant helpers take/return bf16-dequantized values.
"""
import torch

FP8_MAX = 448.0
FP4_MAX = 6.0


# --------------------------------------------------------------------------- exact grid rounding (fp32 math)
def _exp2_floor(a: torch.Tensor) -> torch.Tensor:
    """floor(log2(a)) for a > 0 (fp32), via the IEEE exponent field (exact at powers of two)."""
    return ((a.float().view(torch.int32) >> 23) & 0xFF) - 127


def round_to_e4m3(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values (|v| <= 448 assumed after clamp) to the OCP E4M3 grid, RNE. Returns fp32."""
    a = v.abs()
    e = torch.clamp(_exp2_floor(torch.where(a > 0, a, torch.ones_like(a))), min=-6)  # subnormal quantum 2^-9
    q = torch.pow(2.0, (e - 3).float())
    r = torch.round(a / q) * q
    r = torch.clamp(r, max=FP8_MAX)
    return torch.where(a > 0, torch.copysign(r, v), torch.zeros_like(v))


def round_to_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to the FP4 E2M1 grid {0,.5,1,1.5,2,3,4,6}, RNE, saturating at 6. Returns fp32."""
    a = v.abs()
    e = torch.clamp(_exp2_floor(torch.where(a > 0, a, torch.ones_like(a))), min=0)  # subnormal quantum 0.5
    q = torch.pow(2.0, (e - 1).float())
    r = torch.clamp(torch.round(a / q) * q, max=FP4_MAX)
    return torch.where(a > 0, torch.copysign(r, v), torch.zeros_like(v))


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) for x > 0 (the reference's fast_round_scale)."""
    bits = x.float().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return torch.pow(2.0, e.float())


# --------------------------------------------------------------------------- byte decoders (LUT)
def _e4m3_table() -> torch.Tensor:
    t = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    return torch.nan_to_num(t, nan=0.0)


def _e2m1_table() -> torch.Tensor:
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


_LUTS: dict = {}


def _lut(kind: str, device) -> torch.Tensor:
    key = (kind, str(device))
    if key not in _LUTS:
        # Build on the CPU even under `with torch.device("hpu")`: computed on the HPU the table has inf entries
        # (inexact integer/exp2 arithmetic there), which turned every engram row into inf.
        with torch.device("cpu"):
            table = _e4m3_table() if kind == "e4m3" else _e2m1_table()
        _LUTS[key] = table.to(torch.bfloat16).to(device)
    return _LUTS[key]


def decode_e4m3(b: torch.Tensor) -> torch.Tensor:
    """uint8 E4M3 bytes -> bf16 values (exact)."""
    return _lut("e4m3", b.device)[b.int()]


def decode_e8m0(b: torch.Tensor) -> torch.Tensor:
    """uint8 E8M0 bytes -> fp32 powers of two (exact)."""
    return torch.pow(2.0, b.float() - 127.0)


def decode_e2m1_packed(b: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K//2] packed FP4 (low nibble first) -> bf16 [..., K]."""
    lut = _lut("e2m1", b.device)
    bi = b.int()
    lo, hi = lut[bi & 0x0F], lut[(bi >> 4) & 0x0F]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def dequant_fp4_weight(w_u8: torch.Tensor, s_u8: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP4 weight [N, K//2] + E8M0 scale [N, K//block] -> bf16 [N, K] (exact)."""
    w = decode_e2m1_packed(w_u8)
    s = decode_e8m0(s_u8).to(torch.bfloat16)
    return (w.unflatten(-1, (-1, block)) * s.unsqueeze(-1)).flatten(-2)


def dequant_fp8_weight(w_u8: torch.Tensor, s_u8: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP8 weight [N, K] + E8M0 block scale [ceil(N/b), ceil(K/b)] -> bf16 [N, K] (exact)."""
    N, K = w_u8.shape
    w = decode_e4m3(w_u8)
    s = decode_e8m0(s_u8).to(torch.bfloat16)
    s = s.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w * s


# --------------------------------------------------------------------------- activation quantization
def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None, scale_dtype=torch.float32,
              inplace: bool = False):
    """Block-wise FP8 (E4M3) quantization of the last dim. Returns the dequantized values (input dtype),
    i.e. x_q * s, either in place or as (x_deq, s_fp32) when inplace=False."""
    shape = x.shape
    xg = x.float().unflatten(-1, (-1, block_size))
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = round_to_e4m3(torch.clamp(xg / s, -FP8_MAX, FP8_MAX))
    deq = (q * s).flatten(-2).to(x.dtype).view(shape)
    if inplace:
        x.copy_(deq)
        return x
    return deq, s.squeeze(-1)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False, scale_dtype=None):
    """Block-wise FP4 fake quantization; the reference only uses inplace=True.
    scale_dtype=torch.float8_e4m3fn selects E4M3 scales (compressed KV), else power-of-two (E8M0) scales."""
    shape = x.shape
    xg = x.float().unflatten(-1, (-1, block_size))
    amax = xg.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = round_to_e4m3(amax.clamp_min(6 * 2.0 ** -9) / FP4_MAX)
    else:
        s = pow2_ceil(amax.clamp_min(6 * 2.0 ** -126) / FP4_MAX)
    q = round_to_e2m1(torch.clamp(xg / s, -FP4_MAX, FP4_MAX))
    deq = (q * s).flatten(-2).to(x.dtype).view(shape)
    if inplace:
        x.copy_(deq)
        return x
    return deq, s.squeeze(-1)


# --------------------------------------------------------------------------- attention + hyper-connections
def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float, q_chunk: int = 256) -> torch.Tensor:
    """q [b,m,h,d] bf16, kv [b,n,d] (shared K=V), attn_sink [h], topk_idxs [b,m,topk] (-1 = invalid).
    softmax over the selected keys plus a per-head sink term in the denominator; empty rows -> 0."""
    b, m, h, d = q.shape
    topk_idxs = topk_idxs.to(q.device)  # model.get_window_topk_idxs builds them on the default (CPU) device
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h)
    bidx = torch.arange(b, device=q.device).view(b, 1, 1)
    for s0 in range(0, m, q_chunk):
        s1 = min(m, s0 + q_chunk)
        idx = topk_idxs[:, s0:s1].long()
        valid = idx >= 0
        g = kv[bidx, idx.clamp_min(0)]  # [b, c, topk, d] bf16
        g = g * valid.unsqueeze(-1).to(g.dtype)
        sc = torch.einsum("bchd,bctd->bcht", q[:, s0:s1], g).float() * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = torch.maximum(sc.amax(-1), torch.full_like(sink.expand(b, s1 - s0, h), -1e30))
        p = torch.exp(sc - mx.unsqueeze(-1))
        denom = p.sum(-1) + torch.exp(sink - mx)
        o = torch.einsum("bcht,bctd->bchd", p.to(torch.bfloat16), g).float() / denom.unsqueeze(-1)
        out[:, s0:s1] = o.to(q.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    b, s, _ = mixes.shape
    hc = hc_mult
    m = mixes.reshape(-1, (2 + hc) * hc).float()
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[:, :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[:, hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[:, 2 * hc:] * sc[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)
