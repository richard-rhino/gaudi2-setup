# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (``Qwen4ExpForConditionalGeneration``) for Intel Gaudi (HPU), text only.

Upstream ``vllm.models.qwen4_exp`` only has ``nvidia/`` (Triton/CuTe) and ``amd/`` (Triton/aiter)
backends; on the OOT HPU platform it falls through to ``nvidia/`` and fails at the first kernel.
This is a self-contained port that builds the network from platform-generic vLLM layers plus
vllm_gaudi HPU ops, with the model-specific pieces written in plain torch from the HF reference
(``transformers.models.qwen4_exp.modeling_qwen4_exp``):

* Gated DeltaNet (36 layers): vllm_gaudi ``HPUGatedDeltaNetAttention`` (sigmoid output gate).
* QSA indexed attention (12 layers): runs **dense** gated attention (``Qwen3NextAttention``).
  The QSA indexer keeps every visible token while a query sees <= ``indexer_budget`` complete
  blocks, so dense attention is exact up to ~2k context and a superset of the selection beyond.
* Hyper-connections (``GatedResidual``, hc_count streams): plain torch.
* PLE (layer 2): n-gram ids are hashed **on the CPU** because Gaudi int64 multiply/xor/remainder
  are silently inexact; the 102 GB n-gram table is vocab-sharded across TP ranks. The dilated
  short-conv history and the last ``ngram_size - 1`` token ids live in a per-request state slot that
  borrows the GDN state spec (same shapes/dtypes), so the HPU runner allocates and indexes it like
  one more GDN layer.
* MoE: ``Qwen3NextSparseMoeBlock`` (HPU fused MoE).
"""
import math
import os
import time
from collections.abc import Iterable
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE, SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock
from vllm.distributed import get_pp_group
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                                              extract_layer_index, make_layers)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator, MambaStateShapeCalculator

from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention

_ATTN_TYPES = ("full_attention", "indexed_attention")


_PLE_BYPASS = os.environ.get("QWEN4_PLE_BYPASS") == "1"  # timing diagnostic, wrong outputs


def _ple_host(vllm_config: VllmConfig) -> bool:
    """Keep the PLE n-gram table in host RAM: QWEN4_PLE_HOST=1/0, default on with pipeline parallelism (one stage
    would otherwise need the whole 102 GB table in HBM)."""
    v = os.environ.get("QWEN4_PLE_HOST", "auto")
    return vllm_config.parallel_config.pipeline_parallel_size > 1 if v == "auto" else v == "1"

class GroupedGemmaRMSNorm(nn.Module):
    """RMSNorm over each ``group_size`` slice with Gemma-style ``(1 + w)`` scaling, fp32 math."""

    def __init__(self, dim: int, group_size: int | None, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size, self.eps = group_size, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        if self.group_size is not None:
            xf = xf.unflatten(-1, (-1, self.group_size))
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            xf = xf.flatten(-2)
        return (xf * (1.0 + self.weight.float())).to(x.dtype)


class GatedResidual(nn.Module):
    """Hyper-connection mixer: hc_count residual streams -> one block input (+ injection weights)."""

    def __init__(self, config, use_combine: bool = True):
        super().__init__()
        self.hc, self.H = config.hc_count, config.hidden_size
        hc_hidden = self.hc * self.H
        self.hc_norm = GroupedGemmaRMSNorm(hc_hidden, self.H, config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden, self.hc, bias=False) if use_combine else None

    def forward(self, hyper_input: torch.Tensor):
        normed = self.hc_norm(hyper_input)
        w = F.silu(self.input_mix_weight_down(normed) / self.hc)
        w = torch.sigmoid(self.input_mix_weight_up(w)).unflatten(-1, (self.hc, self.H))
        mixed = (w * normed.unflatten(-1, (self.hc, self.H))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        inj = 2 * torch.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper_input, inj

    @staticmethod
    def combine(hyper_input: torch.Tensor, block_out: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
        return hyper_input + (block_out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)


# ---------------------------------------------------------------- PLE n-gram hashing (CPU, exact int64)
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _SPLITMIX_GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _SPLITMIX_M1) & _MASK64
    v = ((v ^ (v >> 27)) * _SPLITMIX_M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _layer_multipliers(unigram_vocab: int, ngram_size: int, ple_index: int, seed: int) -> list[int]:
    half = max(1, (((1 << 63) - 1) // max(unigram_vocab, 1)) // 2)
    base = seed + _PRIME_1 * ple_index
    return [2 * (_splitmix64((base + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram_size)]


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


from vllm_gaudi.extension.logger import logger as _init_gaudi_logger

logger = _init_gaudi_logger()
_PLE_TIMING = os.getenv("VLLM_PP_STAGE_TIMING", "0") == "1"
_LAYER_TIMING = os.getenv("QWEN4_LAYER_TIMING", "0") == "1"  # per-layer device events (adds ~0.4 ms per layer)
_ple_tacc: dict = {}


def _ple_tick(name, t0):
    """VLLM_PP_STAGE_TIMING=1: accumulate host time of the PLE prep phases, log means every 100 calls."""
    t1 = time.perf_counter()
    n, v = _ple_tacc.get(name, (0, 0.0))
    _ple_tacc[name] = (n + 1, v + t1 - t0)
    if name == "total" and (n + 1) % 100 == 0:
        logger.info("PLE prep host ms: %s", {k: round(1e3 * v / c, 3) for k, (c, v) in _ple_tacc.items()})
        _ple_tacc.clear()
    return t1


class NGramHasher:
    """CPU implementation of HF ``Qwen4ExpTextNGramEmbedding`` id computation (bit-exact)."""

    def __init__(self, config, ple_index: int):
        self.n = config.ngram_size
        self.ctx = self.n - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.num_heads = self.ctx * self.heads_per_ngram
        eos = config.eos_token_id
        self.eos = eos[0] if isinstance(eos, list) else eos
        sizes, offsets, total = [], [], 0
        for h in range(self.num_heads):
            size = _nth_prime_after(config.ngram_vocab_size_base - 1, ple_index * self.num_heads + h + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        div = config.make_ngram_vocab_size_divisible_by
        self.padded_vocab = math.ceil(total / div) * div
        cpu = torch.device("cpu")  # model init runs under a default-device=hpu context; int64 hashing must stay on CPU
        self.sizes = torch.tensor(sizes, dtype=torch.long, device=cpu)
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=cpu)
        self.mult = torch.tensor(_layer_multipliers(config.vocab_size, self.n, ple_index, config.seed), dtype=torch.long,
                                 device=cpu)

    def ids(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """tokens [L] (int64, cpu), context [ctx] previous token ids -> ngram ids [L, num_heads]."""
        hist = torch.cat([context, tokens])
        T = hist.shape[0]
        pos = torch.arange(T, device="cpu")
        eos_pos = torch.where(hist == self.eos, pos, -1)
        prev_eos = torch.cat([torch.tensor([-1], device="cpu"), torch.cummax(eos_pos, 0).values[:-1]])
        pos_in_seg = pos - (prev_eos + 1)
        shifted = [hist]
        for s in range(1, self.n):
            src = (pos - s).clamp_min(0)
            valid = (pos_in_seg >= s) & (pos - s >= 0)
            shifted.append(torch.where(valid, hist[src], torch.full_like(hist, self.eos)))
        blocks = []
        for ngram in range(2, self.n + 1):
            a, b = (ngram - 2) * self.heads_per_ngram, (ngram - 1) * self.heads_per_ngram
            mixed = shifted[0] * self.mult[0]
            for p in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[p] * self.mult[p])
            blocks.append(torch.remainder(mixed.unsqueeze(-1), self.sizes[a:b]) + self.offsets[a:b])
        return torch.cat(blocks, dim=-1)[-tokens.shape[0]:]


class ShardedNGramEmbedding(nn.Module):
    """[padded_vocab, head_dim] table split by rows over TP ranks; checkpoint ships it as row shards."""

    def __init__(self, padded_vocab: int, head_dim: int, shard_rows: int, dtype: torch.dtype, host: bool = False):
        super().__init__()
        self.tp, self.rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        self.per_rank = math.ceil(padded_vocab / self.tp)
        self.start = self.rank * self.per_rank
        self.shard_rows = shard_rows
        self.host = host
        if host:
            # Host RAM copy (lookup-only table, 102 GB on the real model): not a parameter, so the runner never moves
            # it to the device. _ple_prepare gathers rows on the CPU, where the n-gram ids are hashed anyway.
            self._host_w = [torch.zeros(self.per_rank, head_dim, dtype=dtype, device="cpu")]
        else:
            self.weight = nn.Parameter(torch.zeros(self.per_rank, head_dim, dtype=dtype), requires_grad=False)

    def table(self) -> torch.Tensor:
        return self._host_w[0] if self.host else self.weight.data

    def gather_cpu(self, ids: torch.Tensor) -> torch.Tensor:
        """Host-mode lookup of this rank's rows (zeros for rows owned by other TP ranks)."""
        local = ids.long() - self.start
        mask = (local >= 0) & (local < self.per_rank)
        out = F.embedding(torch.where(mask, local, torch.zeros_like(local)), self._host_w[0])
        return out * mask.unsqueeze(-1).to(out.dtype) if self.tp > 1 else out

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, w in weights:  # "shard_K.weight"
            k = int(name.split(".")[0].split("_")[1])
            g0, g1 = k * self.shard_rows, k * self.shard_rows + w.shape[0]
            lo, hi = max(g0, self.start), min(g1, self.start + self.per_rank)
            if lo < hi:
                self.table()[lo - self.start:hi - self.start].copy_(w[lo - g0:hi - g0])
            loaded.add("weight")
        return loaded

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        local = ids - self.start
        mask = (local >= 0) & (local < self.per_rank)
        out = F.embedding(torch.where(mask, local, torch.zeros_like(local)), self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        return tensor_model_parallel_all_reduce(out) if self.tp > 1 else out


_PLE_COMPILE = os.environ.get("QWEN4_PLE_COMPILE", "1") == "1"
_ple_math_fn = None


def _ple_math(self, hidden, emb, hist, vm, n: int, L: int):
    """Pure PLE math (no state I/O, no in-place ops): returns (gated, conv out, padded conv input xp)."""
    hc_hidden = self.H * self.hc
    T = hidden.shape[0]
    key = self.norm_key(self.key_proj(emb)).unflatten(-1, (self.hc, self.H))
    value = self.value_proj(emb)
    query = self.norm_query(hidden).unflatten(-1, (self.hc, self.H))
    gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.H)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    # einsum, not sigmoid(gate) * value.unsqueeze(-2): fused with the sqrt/sign chain above, the HPU graph compiler
    # gets that broadcast multiply wrong for T >= ~128 (~bf16-exact here; repro: ports/qwen4_exp/ple_compile_repro.py)
    gated = torch.einsum("tk,th->tkh", torch.sigmoid(gate.squeeze(-1)), value).flatten(-2)
    gated_normed = self.norm_conv(gated)
    if vm is not None:
        gated, gated_normed = gated * vm, gated_normed * vm
    # dilated causal depthwise conv per sequence, with history from the state slot
    x = gated_normed.view(n, L, hc_hidden)
    xp = torch.cat([hist, x], dim=1)  # [n, state_len + L, C]
    w = self.conv1d.weight.view(hc_hidden, self.kernel)  # w[:, j] multiplies x[t - (K-1-j)*dil]
    out = x.new_zeros(n, L, hc_hidden)
    for j in range(self.kernel):
        off = j * self.dil
        out = out + xp[:, off:off + L] * w[:, j]
    return gated, F.silu(out).view(T, hc_hidden), xp


def _ple_math_compiled():
    """QWEN4_PLE_COMPILE=1 (default): the math as its own graph. Compiling the whole PLE forward inside the decoder
    layer gave wrong outputs (FINDINGS 18); state loads/stores (bitcast views, index_copy_) stay eager."""
    global _ple_math_fn
    if _ple_math_fn is None:
        _ple_math_fn = torch.compile(_ple_math, backend="hpu_backend", dynamic=False)
    return _ple_math_fn


class HpuPLELayer(MambaBase, nn.Module):
    """Per-Layer Embedding with a per-request state slot borrowed from the GDN state spec."""

    def __init__(self, config, vllm_config: VllmConfig, gdn: HPUGatedDeltaNetAttention, ple_index: int, prefix: str):
        nn.Module.__init__(self)
        self.H, self.hc = config.hidden_size, config.hc_count
        hc_hidden = self.H * self.hc
        self.ple_index = ple_index
        self.ple_slot = ple_index  # row in PLEStepInputs.ids; reset to the stage-local index under PP
        self.hasher = NGramHasher(config, ple_index)
        emb_dim = config.ple_embed_dim
        num_parts = config.split_ngram_parts
        shard_rows = math.ceil(self.hasher.padded_vocab / num_parts)
        self.ple_embedding = nn.Module()
        self.ple_embedding.ngram_embedding = ShardedNGramEmbedding(self.hasher.padded_vocab,
                                                                   emb_dim // self.hasher.num_heads, shard_rows,
                                                                   vllm_config.model_config.dtype,
                                                                   host=_ple_host(vllm_config))
        self.key_proj = nn.Linear(emb_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(emb_dim, self.H, bias=False)
        eps = config.rms_norm_eps
        self.norm_key = GroupedGemmaRMSNorm(hc_hidden, self.H, eps)
        self.norm_query = GroupedGemmaRMSNorm(hc_hidden, self.H, eps)
        self.norm_conv = GroupedGemmaRMSNorm(hc_hidden, self.H, eps)
        self.kernel = config.ple_conv_kernel_size
        self.dil = config.ngram_size
        self.state_len = (self.kernel - 1) * self.dil
        self.conv1d = nn.Conv1d(hc_hidden, hc_hidden, self.kernel, groups=hc_hidden, dilation=self.dil, bias=False)
        self.ctx = self.hasher.ctx
        # State slot = this layer's (fp32) GDN-spec ssm-state row: [conv history as raw bf16 pairs | token ids fp32].
        # bf16 history is bit-exact (the activations are bf16) and needs half the fp32 elements.
        self._gdn = [gdn]  # list: keep out of nn.Module children
        self.hist_f32 = (self.state_len * hc_hidden + 1) // 2
        self.state_numel = self.hist_f32 + self.ctx
        self.cache_group_idx = None
        self.kv_cache = None
        self.prefix = prefix
        cc = vllm_config.compilation_config
        if prefix in cc.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        cc.static_forward_context[prefix] = self

    # ---- MambaBase: identical spec to the sibling GDN layer so it shares its KV-cache group type
    def get_state_shape(self):
        return self._gdn[0].get_state_shape()

    def get_state_dtype(self):
        return self._gdn[0].get_state_dtype()

    @property
    def mamba_type(self):
        return self._gdn[0].mamba_type

    def check_capacity(self):
        shape = self.get_state_shape()[1]
        cap = math.prod(shape)
        if cap < self.state_numel:
            raise ValueError(f"PLE state ({self.state_numel}) does not fit in the GDN ssm slot ({cap}) at this TP size")

    # ---- per-request state helpers
    def _slots(self, attn_md, attr):
        idx = getattr(attn_md, attr, None)
        if idx is not None and idx.dim() > 1:
            idx = idx.index_select(0, self.cache_group_idx.view(1)).squeeze(0)
        return idx

    @torch._dynamo.disable  # compiled inside the layer graph, the outputs are wrong (FINDINGS 18)
    def forward(self, hidden: torch.Tensor, ple_in: "PLEStepInputs") -> torch.Tensor:
        """hidden [T, hc*H] -> hidden + PLE(hidden). T is the flat (possibly padded) token axis.

        Pure device code (compiled with the decoder layer): the n-gram ids, sequence lengths and new context tokens
        were computed once for all PLE layers by ``HpuQwen4ExpModel._ple_prepare``."""
        if _PLE_BYPASS:  # timing diagnostic only: output is WRONG with this set
            return hidden
        orig_shape = hidden.shape
        hidden = hidden.reshape(-1, orig_shape[-1])
        md = get_forward_context().attn_metadata
        T = hidden.shape[0]
        hc_hidden = self.H * self.hc
        n = ple_in.lens.shape[0]
        L = T // n
        conv_hist = load = store = None
        if md is not None and self.kv_cache is not None:
            is_prompt = bool(getattr(md, "is_prompt", False))
            load, store = self._slots(md, "load_indices_tensor"), self._slots(md, "store_indices_tensor")
            store = load if store is None else store
            state = self.kv_cache[1].view(self.kv_cache[1].shape[0], -1)
            rows = state.index_select(0, load.long())[:, :self.hist_f32]
            keep = (load.reshape(-1) != 0)
            has_init = getattr(md, "has_initial_states_p", None) if is_prompt else None
            if has_init is not None:
                keep = keep & has_init.bool().view(-1)
            rows = torch.where(keep.view(-1, 1), rows, torch.zeros_like(rows))
            conv_hist = rows.contiguous().view(torch.bfloat16)[:, :self.state_len * hc_hidden]
            conv_hist = conv_hist.reshape(n, self.state_len, hc_hidden)

        table = self.ple_embedding.ngram_embedding
        if ple_in.emb is not None:  # host-mode table: rows were gathered on the CPU in _ple_prepare
            emb = ple_in.emb[self.ple_slot]
            emb = (tensor_model_parallel_all_reduce(emb) if table.tp > 1 else emb).flatten(-2)
        else:
            emb = table(ple_in.ids[self.ple_slot]).flatten(-2)
        hist = conv_hist.to(hidden.dtype) if conv_hist is not None else hidden.new_zeros(n, self.state_len, hc_hidden)
        vm = ple_in.valid.reshape(-1, 1).to(hidden.dtype) if ple_in.valid is not None else None
        math_fn = _ple_math_compiled() if _PLE_COMPILE else _ple_math
        gated, out, xp = math_fn(self, hidden, emb, hist, vm, n, L)

        # ---- write back state (last state_len valid positions + last ctx token ids)
        if store is not None:
            if _saves_every_token(store, n, L):  # the window that ends at each token
                new_hist = torch.stack([xp[:, j + 1:j + 1 + self.state_len] for j in range(L)], dim=1)
                rows = n * L
            else:
                gather = ple_in.lens.view(-1, 1) + torch.arange(self.state_len, device=hidden.device).view(1, -1)
                new_hist = torch.gather(xp, 1, gather.unsqueeze(-1).expand(-1, -1, hc_hidden))
                rows = n
            hist_bf16 = new_hist.reshape(rows, -1).to(torch.bfloat16)
            if hist_bf16.shape[1] % 2:
                hist_bf16 = torch.cat([hist_bf16, hist_bf16.new_zeros(rows, 1)], dim=1)
            new_rows = torch.cat([hist_bf16.contiguous().view(torch.float32), ple_in.new_tok], dim=1)
            st = self.kv_cache[1].view(self.kv_cache[1].shape[0], -1)
            st[:, :self.state_numel].index_copy_(0, store.long(), new_rows.to(st.dtype))
        return (hidden + gated + out).view(orig_shape)


class MultiNGramHasher:
    """All PLE layers' n-gram ids in one vectorized CPU pass (same int64 ops as ``NGramHasher``, layer axis first)."""

    def __init__(self, hashers: list[NGramHasher]):
        h0 = hashers[0]
        self.n, self.ctx, self.hpn, self.eos = h0.n, h0.ctx, h0.heads_per_ngram, h0.eos
        self.num_heads = h0.num_heads
        self.mult = torch.stack([h.mult for h in hashers])  # [P, n]
        self.sizes = torch.stack([h.sizes for h in hashers])  # [P, heads]
        self.offsets = torch.stack([h.offsets for h in hashers])

    def ids(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """tokens [L], context [ctx] (int64, cpu) -> [P, L, num_heads]"""
        hist = torch.cat([context, tokens])
        T = hist.shape[0]
        pos = torch.arange(T, device="cpu")
        eos_pos = torch.where(hist == self.eos, pos, -1)
        prev_eos = torch.cat([torch.tensor([-1], device="cpu"), torch.cummax(eos_pos, 0).values[:-1]])
        pos_in_seg = pos - (prev_eos + 1)
        shifted = [hist]
        for s in range(1, self.n):
            src = (pos - s).clamp_min(0)
            valid = (pos_in_seg >= s) & (pos - s >= 0)
            shifted.append(torch.where(valid, hist[src], torch.full_like(hist, self.eos)))
        blocks = []
        for ngram in range(2, self.n + 1):
            a, b = (ngram - 2) * self.hpn, (ngram - 1) * self.hpn
            mixed = shifted[0].unsqueeze(0) * self.mult[:, 0:1]
            for p in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[p].unsqueeze(0) * self.mult[:, p:p + 1])
            blocks.append(torch.remainder(mixed.unsqueeze(-1), self.sizes[:, None, a:b]) + self.offsets[:, None, a:b])
        return torch.cat(blocks, dim=-1)[:, -tokens.shape[0]:]


def _saves_every_token(store: Optional[torch.Tensor], n: int, length: int) -> bool:
    """Speculative decode: the runner passes one store slot per token, flat [n * L], so that rejected drafts can be
    rewound by loading the last kept token's slot. Otherwise there is one slot per sequence."""
    return store is not None and length > 1 and store.numel() == n * length


class PLEStepInputs(NamedTuple):
    ids: torch.Tensor  # [P, T, num_heads] int32 n-gram ids for every PLE layer
    lens: torch.Tensor  # [n] valid tokens per sequence (long)
    new_tok: torch.Tensor  # [n or n * L, ctx] fp32 last ctx token ids (after every token under spec decode)
    valid: Optional[torch.Tensor]  # [n, L] bool prompt padding mask, or None
    emb: Optional[torch.Tensor] = None  # [P, T, num_heads, head_dim] rows gathered on the host (host-mode table)


class HpuQwen4ExpDecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if layer_type == "linear_attention":
            self.linear_attn = HPUGatedDeltaNetAttention(config, vllm_config=vllm_config,
                                                         prefix=f"{prefix}.linear_attn", gqa_interleaved_layout=False)
        elif layer_type in _ATTN_TYPES:
            self.self_attn = Qwen3NextAttention(config, model_config=vllm_config.model_config,
                                                cache_config=vllm_config.cache_config,
                                                quant_config=vllm_config.quant_config, prefix=f"{prefix}.self_attn")
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")
        self.ple = None
        if (self.layer_idx + 1) in (config.ple_layer_ids or []):
            if layer_type != "linear_attention":
                raise NotImplementedError("PLE state borrows the GDN state spec; PLE on an attention layer")
            ple_index = sorted(config.ple_layer_ids).index(self.layer_idx + 1)
            self.ple = HpuPLELayer(config, vllm_config, self.linear_attn, ple_index, prefix=f"{prefix}.ple")
        self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, ple_in: PLEStepInputs) -> torch.Tensor:
        if self.ple is not None:
            hidden = self.ple(hidden, ple_in)
        x, hyper, inj = self.attn_hyper_connection(hidden)
        if self.layer_type == "linear_attention":
            out = self.linear_attn(hidden_states=x)
        else:
            out = self.self_attn(positions=positions, hidden_states=x)
        hidden = GatedResidual.combine(hyper, out, inj)
        x, hyper, inj = self.mlp_hyper_connection(hidden)
        return GatedResidual.combine(hyper, self.mlp(x), inj)


class HpuQwen4ExpModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        pp = get_pp_group()
        # Pipeline parallel: each stage builds only its own decoder layers (PPMissingLayer elsewhere); the
        # hyper-connection streams [*, hc * hidden] travel between stages as IntermediateTensors.
        self.embed_tokens = (VocabParallelEmbedding(config.vocab_size, config.hidden_size)
                             if pp.is_first_rank else PPMissingLayer())
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HpuQwen4ExpDecoderLayer(vllm_config, config.layer_types[extract_layer_index(prefix)],
                                                   prefix=prefix),
            prefix=f"{prefix}.layers")
        self.hyper_connection_mixer = (GatedResidual(config, use_combine=False)
                                       if pp.is_last_rank else PPMissingLayer())
        local = self.layers[self.start_layer:self.end_layer]
        self.ple_layers = [layer.ple for layer in local if layer.ple is not None]  # plain list, not children
        for j, p in enumerate(self.ple_layers):
            p.check_capacity()
            p.ple_slot = j
        self.ple_hasher = MultiNGramHasher([p.hasher for p in self.ple_layers]) if self.ple_layers else None

    @torch.no_grad()
    def _ple_prepare(self, input_ids: torch.Tensor) -> Optional[PLEStepInputs]:
        """Once per forward (eager): read the context tokens from the first PLE layer's state slot (all PLE layers
        store the same tokens), hash on the CPU for every PLE layer at once and ship the ids to the device.
        Two host syncs per step in total instead of ~3 per PLE layer."""
        if self.ple_hasher is None:
            return None
        hs, p0 = self.ple_hasher, self.ple_layers[0]
        dev = input_ids.device
        t_start = t = time.perf_counter() if _PLE_TIMING else 0.0
        ids_flat = input_ids.reshape(-1)
        T = ids_flat.shape[0]
        md = get_forward_context().attn_metadata
        valid = None
        every_token = False
        if md is None or p0.kv_cache is None:  # profiling / dummy run
            n = 1
            ctx_cpu = torch.full((1, hs.ctx), hs.eos, dtype=torch.long)
            lens = [T]
        else:
            is_prompt = bool(getattr(md, "is_prompt", False))
            load = p0._slots(md, "load_indices_tensor")
            n = int(load.numel())
            L = T // n
            state = p0.kv_cache[1].view(p0.kv_cache[1].shape[0], -1)
            tok = state.index_select(0, load.long())[:, p0.hist_f32:p0.state_numel]
            keep = (load.reshape(-1) != 0)
            has_init = getattr(md, "has_initial_states_p", None) if is_prompt else None
            if has_init is not None:
                keep = keep & has_init.bool().view(-1)
            tok = torch.where(keep.view(-1, 1), tok, torch.full_like(tok, float(hs.eos)))
            pm = getattr(md, "padding_mask_flat", None)
            if is_prompt and pm is not None and pm.numel() == T:
                valid = pm.view(n, L).bool()
                lens_dev = valid.sum(-1).to(tok.dtype).view(-1, 1)
                packed = torch.cat([tok, lens_dev], dim=1).cpu()  # one sync for tokens + lengths
                ctx_cpu, lens = packed[:, :-1].round().long(), packed[:, -1].long().tolist()
            else:
                ctx_cpu, lens = tok.cpu().round().long(), [L] * n
            if not is_prompt:
                every_token = _saves_every_token(p0._slots(md, "store_indices_tensor"), n, L)
        if _PLE_TIMING:
            t = _ple_tick("state_to_cpu", t)
        L = T // n
        ids_cpu = ids_flat.cpu().long()
        if _PLE_TIMING:
            t = _ple_tick("ids_to_cpu", t)
        all_ids = torch.zeros(len(self.ple_layers), T, hs.num_heads, dtype=torch.int32)
        new_tok = torch.empty(n * L if every_token else n, hs.ctx, dtype=torch.float32)
        for i in range(n):
            nv = int(lens[i])
            toks = ids_cpu[i * L:i * L + nv]
            if nv > 0:  # warmup pads with all-padding sequences
                all_ids[:, i * L:i * L + nv] = hs.ids(toks, ctx_cpu[i]).to(torch.int32)
            seen = torch.cat([ctx_cpu[i], toks])
            if every_token:
                new_tok[i * L:(i + 1) * L] = seen.unfold(0, hs.ctx, 1)[1:].to(torch.float32)
            else:
                new_tok[i] = seen[-hs.ctx:].to(torch.float32)
        if _PLE_TIMING:
            t = _ple_tick("hash", t)
        emb = None
        if self.ple_layers[0].ple_embedding.ngram_embedding.host:
            emb = torch.stack([p.ple_embedding.ngram_embedding.gather_cpu(all_ids[j])
                               for j, p in enumerate(self.ple_layers)]).to(dev, non_blocking=True)
        if _PLE_TIMING:
            t = _ple_tick("gather_h2d", t)
        out = PLEStepInputs(all_ids.to(dev, non_blocking=True), torch.tensor(lens, dtype=torch.long).to(dev),
                            new_tok.to(dev), valid, emb)
        if _PLE_TIMING:
            _ple_tick("other_h2d", t)
            _ple_tick("total", t_start)
        return out

    def _layer_events(self, ev):
        """VLLM_PP_STAGE_TIMING=1: device ms of [PLE prep, layer start..end) for single-token decode, every 100."""
        pend = self.__dict__.setdefault("_ev_pend", [])
        pend.append(ev)
        if len(pend) < 100:
            return
        pend[-1][-1].synchronize()
        n = len(pend[0]) - 1
        means = [sum(e[i].elapsed_time(e[i + 1]) for e in pend) / len(pend) for i in range(n)]
        pend.clear()
        logger.info("stage layers %d-%d device ms: prep %.2f, layers %s", self.start_layer, self.end_layer, means[0],
                    [round(m, 2) for m in means[1:]])

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        pp = get_pp_group()
        if pp.is_first_rank:
            h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            h = h.repeat(*([1] * (h.dim() - 1)), self.config.hc_count)  # HPU passes [B, L] or [T] inputs
        else:
            h = intermediate_tensors["hidden_states"]
        if _LAYER_TIMING:
            ev = [torch.hpu.Event(enable_timing=True)]
            ev[0].record()
        ple_in = self._ple_prepare(input_ids)
        if _LAYER_TIMING:
            ev.append(torch.hpu.Event(enable_timing=True))
            ev[-1].record()
        for layer in self.layers[self.start_layer:self.end_layer]:
            h = layer(h, positions, ple_in)
            if _LAYER_TIMING:
                ev.append(torch.hpu.Event(enable_timing=True))
                ev[-1].record()
        if _LAYER_TIMING and input_ids.numel() == 1:
            self._layer_events(ev)
        if not pp.is_last_rank:
            return IntermediateTensors({"hidden_states": h})
        return self.hyper_connection_mixer(h)


class HpuQwen4ExpForConditionalGeneration(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE, SupportsPP):
    """Text-only HPU implementation registered under the upstream multimodal architecture name."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.language_model.": "model.",
        "model.visual.": None,
        "mtp.": None,
    }) | Qwen3_5Model.hf_to_vllm_mapper | WeightsMapper(orig_to_new_substr={
        ".self_attn.indexer.": None,  # QSA indexer unused: dense attention
        ".ple.ple_embedding.layer_multipliers": None,  # recomputed (and checked) by NGramHasher
        ".ple.ple_embedding.ngram_heads_offsets": None,
        ".ple.ple_embedding.ngram_heads_vocab_sizes": None,
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = vllm_config.model_config.hf_config  # runner reads .architectures from the top-level config
        self.vllm_config = vllm_config
        self.model = HpuQwen4ExpModel(vllm_config=vllm_config, prefix="model")
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head")
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

    def make_empty_intermediate_tensors(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        c = self.vllm_config.model_config.hf_text_config
        return IntermediateTensors({"hidden_states": torch.zeros(batch_size, c.hc_count * c.hidden_size,
                                                                 dtype=dtype, device=device)})

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds, intermediate_tensors)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    # ---- hybrid state hooks (same as upstream qwen3_5: GDN state only; PLE borrows the GDN spec)
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(vllm_config.model_config.dtype,
                                                                     vllm_config.cache_config.mamba_cache_dtype,
                                                                     vllm_config.cache_config.mamba_ssm_cache_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        c = vllm_config.model_config.hf_text_config
        tp = vllm_config.parallel_config.tensor_parallel_size
        return MambaStateShapeCalculator.gated_delta_net_state_shape(tp, c.linear_num_key_heads,
                                                                     c.linear_num_value_heads, c.linear_key_head_dim,
                                                                     c.linear_value_head_dim, c.linear_conv_kernel_dim,
                                                                     0)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
