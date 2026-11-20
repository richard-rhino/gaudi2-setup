# Pure-PyTorch drop-in replacement for kernel.py (TileLang version).
# All math is done in float32 on the target device; fp8/fp4 dtypes are only
# used for storage casts. No CUDA/triton/tilelang dependencies.

import torch
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers: power-of-two scale rounding (MXFP-style)
# ---------------------------------------------------------------------------

def _fast_log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) for positive float32 tensors, via IEEE-754 bit tricks."""
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    return (exp - 127 + (man != 0).to(torch.int32)).to(torch.int32)


def _fast_pow2(x: torch.Tensor) -> torch.Tensor:
    """2^x for integer-valued float32 tensors, via IEEE-754 bit tricks."""
    bits = ((x.to(torch.int32) + 127) << 23).to(torch.int32)
    return bits.view(torch.float32)


def _fast_round_scale(amax: torch.Tensor, fp8_max_inv: float) -> torch.Tensor:
    return _fast_pow2(_fast_log2_ceil(amax * fp8_max_inv).to(torch.float32))


# ---------------------------------------------------------------------------
# FP4 e2m1 nibble packing / unpacking (little-nibble-first along last dim)
# ---------------------------------------------------------------------------

# e2m1 representable magnitudes (sign handled separately)
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _fp4_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode float32 tensor (already clamped to [-6, 6]) into e2m1 nibbles (uint8, 0..15)."""
    sign = (x < 0).to(torch.uint8)
    ax = x.abs()
    # Round-to-nearest-even against the 8 representable magnitudes.
    vals = _E2M1_VALUES.to(x.device)
    # Find nearest index via argmin of |ax - v|; ties resolved by lower index (RNE-ish).
    diff = (ax.unsqueeze(-1) - vals).abs()
    idx = diff.argmin(dim=-1).to(torch.uint8)
    return (sign << 3) | idx


def _fp4_decode(nib: torch.Tensor) -> torch.Tensor:
    """Decode e2m1 nibbles (uint8 0..15) to float32."""
    sign = (nib >> 3) & 1
    idx = (nib & 0x7).long()
    vals = _E2M1_VALUES.to(nib.device)[idx]
    return torch.where(sign.bool(), -vals, vals)


def _pack_fp4(nib: torch.Tensor) -> torch.Tensor:
    """Pack nibbles (uint8, last dim even) into bytes: low nibble = even index."""
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack bytes into nibbles (uint8), low nibble first."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty(
        packed.shape[:-1] + (packed.shape[-1] * 2,),
        dtype=torch.uint8, device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _fp4_storage_to_float(packed: torch.Tensor) -> torch.Tensor:
    """Convert a float4_e2m1fn_x2 storage tensor to float32 (unpacked)."""
    raw = packed.view(torch.uint8)
    nib = _unpack_fp4(raw)
    return _fp4_decode(nib)


def _float_to_fp4_storage(x: torch.Tensor) -> torch.Tensor:
    """Convert float32 tensor to float4_e2m1fn_x2 storage (packed along last dim)."""
    nib = _fp4_encode(x)
    packed = _pack_fp4(nib)
    return packed.view(torch.float4_e2m1fn_x2)


# ---------------------------------------------------------------------------
# act_quant: block-wise FP8 quantization
# ---------------------------------------------------------------------------

def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16.
    When scale_fmt is set, scales are rounded to power-of-2 (MXFP)."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    orig_shape = z.shape
    M = z.numel() // N
    xf = z.view(M, N).to(torch.float32)

    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    fp8_min = -448.0

    xg = xf.view(M, N // block_size, block_size)
    amax = xg.abs().amax(dim=-1)  # [M, N//block]
    amax = torch.clamp(amax, min=1e-4)

    if scale_fmt is not None:
        s = _fast_round_scale(amax, fp8_max_inv)
    else:
        s = amax * fp8_max_inv

    # Quantize
    q = torch.clamp(xg / s.unsqueeze(-1), fp8_min, fp8_max)
    q_fp8 = q.to(torch.float8_e4m3fn)  # storage cast

    if inplace:
        # Dequantize back to input dtype
        deq = q_fp8.to(torch.float32) * s.unsqueeze(-1)
        y = deq.view(M, N).to(z.dtype).view(orig_shape)
        x.copy_(y)
        return x

    y = q_fp8.view(orig_shape)
    s_out = s.to(scale_dtype).view(*orig_shape[:-1], N // block_size)
    return y, s_out


# ---------------------------------------------------------------------------
# fp4_act_quant: block-wise FP4 quantization
# ---------------------------------------------------------------------------

def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    """FP4 with E8M0 scales for the indexer or E4M3 scales for compressed KV.
    inplace=True writes the dequantized values back to x."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    orig_shape = z.shape
    M = z.numel() // N
    xf = z.view(M, N).to(torch.float32)

    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max

    xg = xf.view(M, N // block_size, block_size)
    amax = xg.abs().amax(dim=-1)

    if scale_dtype == torch.float8_e4m3fn:
        # E4M3 scale: keep even all-zero group's scale nonzero.
        amax = torch.clamp(amax, min=6.0 * (2.0 ** -9))
        s = (amax / fp4_max).to(torch.float8_e4m3fn).to(torch.float32)
    else:
        amax = torch.clamp(amax, min=6.0 * (2.0 ** -126))
        s = _fast_round_scale(amax, fp4_max_inv)

    q = torch.clamp(xg / s.unsqueeze(-1), -fp4_max, fp4_max)

    if inplace:
        # Quantize to fp4 then dequantize back
        nib = _fp4_encode(q.reshape(-1, block_size))
        deq = _fp4_decode(nib).view(M, N // block_size, block_size) * s.unsqueeze(-1)
        y = deq.view(M, N).to(z.dtype).view(orig_shape)
        x.copy_(y)
        return x

    # Pack to float4_e2m1fn_x2 storage
    nib = _fp4_encode(q.reshape(-1, block_size))
    packed = _pack_fp4(nib)  # [M*N//block, block//2]
    y = packed.view(torch.float4_e2m1fn_x2).view(*orig_shape[:-1], N // 2)
    s_out = s.to(scale_dtype).view(*orig_shape[:-1], N // block_size)
    return y, s_out


# ---------------------------------------------------------------------------
# fp8_gemm: C = A @ B^T with per-block FP8 scaling
# ---------------------------------------------------------------------------

def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-block FP8 scaling."""
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    assert block_size in (32, 128)
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % block_size == 0
    assert a_s.numel() == M * (K // block_size)
    assert b_s.shape == ((N + block_size - 1) // block_size, K // block_size)

    # Dequantize to float32
    af = a.view(M, K).to(torch.float32)
    bf = b.view(N, K).to(torch.float32)
    a_sf = a_s.view(M, K // block_size).to(torch.float32)
    b_sf = b_s.to(torch.float32)

    # Apply scales per K-block
    af = af.view(M, K // block_size, block_size) * a_sf.unsqueeze(-1)
    af = af.view(M, K)
    # B scales are per (N-block, K-block); expand to per-row
    b_row_scale = b_sf.repeat_interleave(block_size, dim=0)[:N]  # [N, K//block]
    bf = bf.view(N, K // block_size, block_size) * b_row_scale.unsqueeze(-1)
    bf = bf.view(N, K)

    c = (af @ bf.t()).to(torch.get_default_dtype())
    return c.view(*a.shape[:-1], N)


# ---------------------------------------------------------------------------
# sparse_attn: gather top-k KV + online softmax with attn sink
# ---------------------------------------------------------------------------

def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, s, h, d = q.size()
    orig_h = h
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
        h = 16

    topk = topk_idxs.size(-1)
    qf = q.to(torch.float32)          # [b, s, h, d]
    kvf = kv.to(torch.float32)        # [b, n, d]
    sinkf = attn_sink.to(torch.float32)  # [h]

    # Gather KV: [b, s, topk, d], invalid (-1) -> 0
    valid = topk_idxs != -1  # [b, s, topk]
    idx_safe = topk_idxs.clamp(min=0)  # [b, s, topk]
    # kv[b, idx_safe, :] -> [b, s, topk, d]
    b_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, s, topk)
    gathered = kvf[b_idx, idx_safe]  # [b, s, topk, d]
    gathered = gathered * valid.unsqueeze(-1).to(torch.float32)

    # Scores: [b, s, h, topk]
    scores = torch.einsum("bshd,bstd->bsht", qf, gathered) * softmax_scale
    # Mask invalid positions to -inf
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # Online softmax with finite lower bound (matches kernel: -1e30)
    # Use a stable softmax over valid entries; invalid entries contribute 0.
    # scores_max starts at -1e30 (finite) so all-invalid rows give 0 output.
    NEG = -1e30
    scores_max = scores.amax(dim=-1)  # [b, s, h]
    scores_max = torch.maximum(scores_max, torch.full_like(scores_max, NEG))

    # exp(scores - max); invalid -> exp(-inf - finite) = 0
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))
    # For all-invalid rows, scores_max = NEG, scores = -inf -> exp(-inf - NEG) = 0
    exp_scores = torch.nan_to_num(exp_scores, nan=0.0, posinf=0.0, neginf=0.0)

    sum_exp = exp_scores.sum(dim=-1)  # [b, s, h]
    # Add attn sink contribution
    sum_exp = sum_exp + torch.exp(sinkf - scores_max)  # [b, s, h]

    # Output: [b, s, h, d]
    o = torch.einsum("bsht,bstd->bshd", exp_scores, gathered)
    o = o / sum_exp.unsqueeze(-1)

    o = o.to(q.dtype)
    if orig_h < 16:
        o = o.narrow(2, 0, orig_h).contiguous()
    return o


# ---------------------------------------------------------------------------
# hc_split_sinkhorn: split mixes into pre/post/comb with Sinkhorn normalization
# ---------------------------------------------------------------------------

def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.size()
    hc = hc_mult
    mix_hc = (2 + hc) * hc

    m = mixes.view(-1, mix_hc).to(torch.float32)
    scale = hc_scale.to(torch.float32)
    base = hc_base.to(torch.float32)

    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])

    comb = m[:, 2 * hc:].view(-1, hc, hc) * scale[2] + base[2 * hc:].view(hc, hc)

    # comb = comb.softmax(-1) + eps
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps

    # comb = comb / (comb.sum(-2) + eps)
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)

    pre = pre.view(b, s, hc)
    post = post.view(b, s, hc)
    comb = comb.view(b, s, hc, hc)
    return pre, post, comb


# ---------------------------------------------------------------------------
# fp4_gemm: C = A_fp8 @ B_fp4^T
# ---------------------------------------------------------------------------

def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T.
    A has per-32 or per-128 activation scale; B has per-32 E8M0 weight scale.
    B is stored as [N, K//2] in float4_e2m1fn_x2 (2 FP4 values per byte, packed along K)."""
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert act_block_size in (32, 128)
    assert K % act_block_size == 0
    assert a_s.numel() == M * (K // act_block_size)
    assert b_s.shape == (N, K // 32)

    # Dequantize A (fp8) with per-act-block scales
    af = a.view(M, K).to(torch.float32)
    a_sf = a_s.view(M, K // act_block_size).to(torch.float32)
    af = af.view(M, K // act_block_size, act_block_size) * a_sf.unsqueeze(-1)
    af = af.view(M, K)

    # Dequantize B (fp4) with per-32 scales
    bf = _fp4_storage_to_float(b.view(N, K // 2))  # [N, K]
    b_sf = b_s.to(torch.float32)  # [N, K//32]
    bf = bf.view(N, K // 32, 32) * b_sf.unsqueeze(-1)
    bf = bf.view(N, K)

    c = (af @ bf.t()).to(torch.get_default_dtype())
    return c.view(*a.shape[:-1], N)
