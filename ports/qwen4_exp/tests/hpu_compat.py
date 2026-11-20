"""Small HPU workarounds for ops that don't compile on Gaudi2 (Gaudi SW 1.24.2)."""
import torch
_orig_cummax = torch.cummax

def cummax_logstep(x: torch.Tensor, dim: int):
    """Running max via log2(L) shift+max steps. Returns (values, None) - indices unsupported."""
    dim = dim % x.dim()
    L = x.shape[dim]
    v = x
    s = 1
    while s < L:
        pad_shape = list(v.shape); pad_shape[dim] = s
        fill = torch.full(pad_shape, torch.iinfo(v.dtype).min if not v.dtype.is_floating_point else float("-inf"),
                          dtype=v.dtype, device=v.device)
        shifted = torch.cat([fill, v.narrow(dim, 0, L - s)], dim=dim)
        v = torch.maximum(v, shifted)
        s *= 2
    class R: pass
    r = R(); r.values = v; r.indices = None
    return r

def _cummax(x, dim, *a, **k):
    if x.device.type == "hpu":
        return cummax_logstep(x, dim)
    return _orig_cummax(x, dim, *a, **k)

def install():
    torch.cummax = _cummax

_orig_solve_tri = torch.linalg.solve_triangular

def unitri_lower_inverse(A: torch.Tensor) -> torch.Tensor:
    """Exact inverse of a unit-lower-triangular matrix via (I-N)(I+N^2)(I+N^4)... (N nilpotent). Matmul-only."""
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    N = torch.tril(A, diagonal=-1)
    P = -N                      # (-N)^(2^k)
    inv = eye + P
    k = 2
    while k < n:
        P = P @ P
        inv = inv @ (eye + P)
        k *= 2
    return inv

def _solve_tri(A, B, *, upper, left=True, unitriangular=False, out=None):
    if A.device.type == "hpu" and unitriangular and not upper and left and out is None:
        dt = B.dtype
        return (unitri_lower_inverse(A.float()) @ B.float()).to(dt)
    return _orig_solve_tri(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)

def install_all():
    install()
    torch.linalg.solve_triangular = _solve_tri

def patch_hf_ngram_cpu():
    """HF Qwen4Exp n-gram ids need exact int64 math; HPU int64 is inexact. Hash on CPU, embed on device."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    orig_forward = M.Qwen4ExpTextNGramEmbedding.forward
    def forward(self, input_ids, past_key_values):
        dev = input_ids.device
        if dev.type != "hpu":
            return orig_forward(self, input_ids, past_key_values)
        emb = self.ngram_embedding
        if not hasattr(self, "_cpu_bufs"):
            self._cpu_bufs = (self.layer_multipliers.cpu(), self.ngram_heads_vocab_sizes.cpu(), self.ngram_heads_offsets.cpu())
        saved = (self.layer_multipliers, self.ngram_heads_vocab_sizes, self.ngram_heads_offsets)
        self.layer_multipliers, self.ngram_heads_vocab_sizes, self.ngram_heads_offsets = self._cpu_bufs
        # run the id computation on CPU with a CPU proxy cache for the token history
        cache = past_key_values
        if cache is not None:
            st = cache.layers[self.layer_idx].conv_states if hasattr(cache.layers[self.layer_idx], "conv_states") else None
        self.ngram_embedding = _IdsOnly()
        try:
            ids = orig_forward(self, input_ids.cpu(), _CpuCacheProxy(cache, self.layer_idx) if cache is not None else None)
        finally:
            self.ngram_embedding = emb
            self.layer_multipliers, self.ngram_heads_vocab_sizes, self.ngram_heads_offsets = saved
        return emb(ids.to(dev).int()).flatten(-2)
    M.Qwen4ExpTextNGramEmbedding.forward = forward

class _IdsOnly(torch.nn.Module):
    """Stand-in for nn.Embedding that returns the ids (shape [..., heads, 1]) so the caller's flatten(-2) is a no-op."""
    weight = torch.empty(0, device="cpu")
    def forward(self, ids):
        return ids.unsqueeze(-1)

class _CpuCacheProxy:
    """Mirror of the HF cache's state_idx=2 token-history slot, kept on CPU to avoid int64 on HPU."""
    _store = {}
    def __init__(self, cache, layer_idx):
        self.cache, self.layer_idx = cache, layer_idx
        self.key = (id(cache), layer_idx)
    def has_previous_state(self, layer_idx, state_idx):
        return self.key in _CpuCacheProxy._store
    @property
    def layers(self):
        proxy = self
        class L: pass
        l = L(); l.conv_states = {2: _CpuCacheProxy._store[proxy.key]}
        return {self.layer_idx: l}
    def update_conv_state(self, x, layer_idx, state_idx, conv_kernel_size):
        prev = _CpuCacheProxy._store.get(self.key)
        hist = x if prev is None else torch.cat([prev, x], dim=-1)
        _CpuCacheProxy._store[self.key] = hist[..., -conv_kernel_size:].clone()
        return None

_orig_scatter = torch.Tensor.scatter

def _scatter(self, dim, index, src=None, *a, value=None, **k):
    if self.device.type == "hpu" and self.dtype == torch.bool:
        v = src if src is not None else value
        if isinstance(v, bool):
            v = int(v)
        elif isinstance(v, torch.Tensor):
            v = v.to(torch.int8)
        return _orig_scatter(self.to(torch.int8), dim, index.int() if index.dtype == torch.long else index, v, *a, **k).bool()
    if src is None and value is not None:
        return _orig_scatter(self, dim, index, value, *a, **k)
    return _orig_scatter(self, dim, index, src, *a, **k)

def install_scatter_bool():
    torch.Tensor.scatter = _scatter
