# SPDX-License-Identifier: Apache-2.0
"""Static-shape decode path for DeepSeek-V4.1's `Attention` and `Indexer`, so that torch.compile can reuse its graphs.

The reference attends over `window + compress_len` keys and `window + min(index_topk, compress_len)` selected slots.
Both grow with the position, so every decode step is a new shape. Here, for a decode step (start_pos > 0, seqlen 1):
* the compressed caches are read up to a power-of-two bucket (>= compress_len, at least BUCKET_MIN);
* positions >= compress_len score -inf and always come out as -1 in the index list, which sparse_attn ignores;
* everything between the cache writes and all-reduces is a pure function of tensors, so the layers share graphs.
Prefill uses the reference code unchanged.

Selection differs from the reference only where it is arbitrary there: ties between equal scores at the top-k edge.
"""
import os

import torch
import torch.nn.functional as F

import kernel_torch as KT
import model as M

BUCKET_MIN = int(os.getenv("DSV41_ATTN_BUCKET_MIN", "1024"))
MASKED_BUT_VISIBLE = -1e38  # below every score, above the -inf of positions that do not exist yet


def bucket(n: int, cap: int) -> int:
    b = BUCKET_MIN
    while b < n:
        b *= 2
    return min(b, cap)


# --------------------------------------------------------------------------- pure pieces
def _norm(x, weight, eps):
    f = x.float()
    f = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
    return (weight * f).to(x.dtype)


def _fp8_linear(x, weight, act_block, scale_fmt):
    xq, _ = KT.act_quant(x, act_block, scale_fmt)
    return F.linear(xq, weight).to(x.dtype)


def _rot_tail(x, freqs, rd, inverse=False):
    """Functional apply_rotary_emb on the last `rd` values. freqs [1, rd/2, 2]."""
    nope, rope = x.split([x.size(-1) - rd, rd], dim=-1)
    a, b = rope.float().unflatten(-1, (-1, 2)).unbind(-1)
    c, s = freqs.unbind(-1)
    if inverse:
        s = -s
    shape = [1, 1] + [1] * (a.ndim - 3) + [-1]
    c, s = c.reshape(shape), s.reshape(shape)
    rot = torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2).to(x.dtype)
    return torch.cat([nope, rot], dim=-1)


def query_kv(x, wa, qn, wb, wkv, kvn, freqs, eps, heads, head_dim, rd, act_block, scale_fmt):
    qr = _norm(_fp8_linear(x, wa, act_block, scale_fmt), qn, eps)
    q = _fp8_linear(qr, wb, act_block, scale_fmt).unflatten(-1, (heads, head_dim))
    kv = _norm(_fp8_linear(x, wkv, act_block, scale_fmt), kvn, eps)
    kv, _ = KT.act_quant(_rot_tail(kv, freqs, rd), act_block, scale_fmt)
    return qr, _rot_tail(q, freqs, rd), kv


def attend(q, window, compressed, sink, idxs, freqs, softmax_scale, rd):
    # Not compiled (see install): through M.sparse_attn, which install_compile compiles on its own.
    keys = torch.cat([window, compressed], dim=1)
    return _rot_tail(M.sparse_attn(q, keys, sink, idxs, softmax_scale), freqs, rd, True)


def project(o, wa, wb, groups, act_block, scale_fmt):
    o, _ = KT.act_quant(o.reshape(o.size(0), 1, groups, -1), act_block, scale_fmt)
    o = torch.einsum("bsgd,grd->bsgr", o, wa).flatten(2)
    return _fp8_linear(o, wb, act_block, scale_fmt).float()


def latent_kv(latent, freqs, rd):
    # compressed KV: groups of 16 with E4M3 scales (the index keys use 32 with E8M0)
    return KT.fp4_act_quant(_rot_tail(latent, freqs, rd), 16, scale_dtype=torch.float8_e4m3fn)[0]


def index_key(latent, wk, kn, freqs, eps, rd, block):
    k = _rot_tail(_norm(F.linear(latent, wk), kn, eps), freqs, rd)
    return KT.fp4_act_quant(k, block)[0]


def index_scores(x, qr, keys, wb, wp, freqs, heads, head_dim, rd, scale, act_block, scale_fmt, block):
    q = _fp8_linear(qr, wb, act_block, scale_fmt).unflatten(-1, (heads, head_dim))
    q = KT.fp4_act_quant(_rot_tail(q, freqs, rd), block)[0]
    weights = F.linear(x, wp) * scale
    score = torch.einsum("bshd,btd->bsht", q, keys)
    return (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=2)


def candidate_mask(score, length, topk_blocks, block):
    pos = torch.arange(score.size(-1), device=score.device)
    return M.select_candidate_blocks(score.masked_fill(pos >= length, -torch.inf), length, topk_blocks, block)


def select_topk(score, length, candidates, k, offset):
    pos = torch.arange(score.size(-1), device=score.device)
    if candidates is not None:
        score = score.masked_fill(~candidates, MASKED_BUT_VISIBLE)
    score = score.masked_fill(pos >= length, -torch.inf)
    idxs = score.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values.to(torch.int32)
    return torch.where(idxs < length, idxs + offset, torch.full_like(idxs, -1)).to(torch.int32)


# --------------------------------------------------------------------------- stateful drivers
_PURE = (query_kv, attend, project, latent_kv, index_key, index_scores, candidate_mask, select_topk)
_FN = {f.__name__: f for f in _PURE}
_orig_forward = None


def _indexer_step(attn, x, qr, latent, start_pos, length, length_t, offset):
    ix = attn.indexer
    bsz, ratio, rd = x.size(0), ix.compress_ratio, ix.rope_head_dim
    if ix.owns_k:
        if latent is not None:
            freqs = attn.freqs_cis[start_pos + 1 - ratio : start_pos + 2 - ratio].clone()
            k = _FN["index_key"](latent, ix.wk.weight, ix.k_norm.weight, freqs, ix.k_norm.eps, rd, M.fp4_block_size)
            ix.k_cache[:bsz, start_pos // ratio : start_pos // ratio + 1] = k
        M.shared_attn.index_k = ix.k_cache
    cache = M.shared_attn.index_k
    width = bucket(length, cache.size(1))
    score = _FN["index_scores"](
        x, qr, cache[:bsz, :width], ix.wq_b.weight, ix.weights_proj.weight,
        attn.freqs_cis[start_pos : start_pos + 1].clone(), ix.n_local_heads, ix.index_head_dim, rd,
        ix.softmax_scale * ix.n_heads**-0.5, M.fp8_block_size, M.scale_fmt, M.fp4_block_size,
    )
    if M.world_size > 1:
        torch.distributed.all_reduce(score)
    candidates = None
    if ix.is_candidate_source:
        M.shared_attn.candidates = _FN["candidate_mask"](
            score, length_t, ix.candidate_topk_blocks, ix.candidate_block_size)
    elif ix.uses_candidates:
        candidates = M.shared_attn.candidates
    return _FN["select_topk"](score, length_t, candidates, min(ix.index_topk, width), offset)


def forward(self, x, start_pos: int):
    if start_pos == 0 or x.size(1) != 1:
        return _orig_forward(self, x, start_pos)
    bsz, win, ratio, rd = x.size(0), self.window_size, self.compress_ratio, self.rope_head_dim
    act_block, scale_fmt = M.fp8_block_size, M.scale_fmt
    freqs = self.freqs_cis[start_pos : start_pos + 1].clone()  # clone: a graph input without a changing view offset
    qr, q, kv = _FN["query_kv"](
        x, self.wq_a.weight, self.q_norm.weight, self.wq_b.weight, self.wkv.weight, self.kv_norm.weight, freqs,
        self.eps, self.n_local_heads, self.head_dim, rd, act_block, scale_fmt,
    )
    self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
    window = self.window_kv_cache[:bsz]
    idxs = M.get_window_topk_idxs(win, bsz, 1, start_pos).to(x.device)
    compressed = window[:, :0]
    if ratio:
        length = (start_pos + 1) // ratio
        length_t = torch.tensor(length, dtype=torch.int32, device=x.device)
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            M.shared_attn.compress_kv = self.compress_kv_cache
        if self.is_index_source:  # before the cache write, as in the reference
            M.shared_attn.topk_idxs = _indexer_step(self, x, qr, latent, start_pos, length, length_t, win)
        if latent is not None:
            lf = self.freqs_cis[start_pos + 1 - ratio : start_pos + 2 - ratio].clone()
            value = _FN["latent_kv"](latent, lf, rd)
            self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + 1] = value
        cache = M.shared_attn.compress_kv
        compressed = cache[:bsz, : bucket(length, cache.size(1))]
        idxs = torch.cat([idxs, M.shared_attn.topk_idxs], dim=-1)
    wa = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = _FN["attend"](q, window, compressed, self.attn_sink, idxs, freqs, self.softmax_scale, rd)
    y = _FN["project"](o, wa, self.wo_b.weight, self.n_local_groups, act_block, scale_fmt)
    if M.world_size > 1:
        torch.distributed.all_reduce(y)
    return y.type_as(x)


def install(compile_pieces: bool):
    global _orig_forward
    if _orig_forward is None:
        _orig_forward = M.Attention.forward
    from hpu_patches import _offset_safe  # same guard as the other compiled pieces
    # `attend` compiled as one graph crashes a later compiled graph on Gaudi 1.24 ("offset_ 0 is not matching with the
    # offset of new tensor ..."; bisected on modules 0-3). DSV41_ATTN_EAGER: more pieces to leave uncompiled.
    eager = {"attend"} | set(filter(None, os.getenv("DSV41_ATTN_EAGER", "").split(",")))
    for f in _PURE:
        _FN[f.__name__] = (_offset_safe(torch.compile(f, backend="hpu_backend", dynamic=False), f.__name__)
                           if compile_pieces and f.__name__ not in eager and "all" not in eager else f)
    M.Attention.forward = forward


def uninstall():
    if _orig_forward is not None:
        M.Attention.forward = _orig_forward
