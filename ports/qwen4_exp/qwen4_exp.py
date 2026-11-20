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
from collections.abc import Iterable

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
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, extract_layer_index
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator, MambaStateShapeCalculator

from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention

_ATTN_TYPES = ("full_attention", "indexed_attention")


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

    def __init__(self, padded_vocab: int, head_dim: int, shard_rows: int, dtype: torch.dtype):
        super().__init__()
        self.tp, self.rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        self.per_rank = math.ceil(padded_vocab / self.tp)
        self.start = self.rank * self.per_rank
        self.shard_rows = shard_rows
        self.weight = nn.Parameter(torch.zeros(self.per_rank, head_dim, dtype=dtype), requires_grad=False)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, w in weights:  # "shard_K.weight"
            k = int(name.split(".")[0].split("_")[1])
            g0, g1 = k * self.shard_rows, k * self.shard_rows + w.shape[0]
            lo, hi = max(g0, self.start), min(g1, self.start + self.per_rank)
            if lo < hi:
                self.weight.data[lo - self.start:hi - self.start].copy_(w[lo - g0:hi - g0])
            loaded.add("weight")
        return loaded

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        local = ids - self.start
        mask = (local >= 0) & (local < self.per_rank)
        out = F.embedding(torch.where(mask, local, torch.zeros_like(local)), self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        return tensor_model_parallel_all_reduce(out) if self.tp > 1 else out


class HpuPLELayer(MambaBase, nn.Module):
    """Per-Layer Embedding with a per-request state slot borrowed from the GDN state spec."""

    def __init__(self, config, vllm_config: VllmConfig, gdn: HPUGatedDeltaNetAttention, ple_index: int, prefix: str):
        nn.Module.__init__(self)
        self.H, self.hc = config.hidden_size, config.hc_count
        hc_hidden = self.H * self.hc
        self.hasher = NGramHasher(config, ple_index)
        emb_dim = config.ple_embed_dim
        num_parts = config.split_ngram_parts
        shard_rows = math.ceil(self.hasher.padded_vocab / num_parts)
        self.ple_embedding = nn.Module()
        self.ple_embedding.ngram_embedding = ShardedNGramEmbedding(self.hasher.padded_vocab,
                                                                   emb_dim // self.hasher.num_heads, shard_rows,
                                                                   vllm_config.model_config.dtype)
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

    @torch._dynamo.disable
    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """hidden [T, hc*H] -> hidden + PLE(hidden). T is the flat (possibly padded) token axis."""
        orig_shape = hidden.shape
        hidden = hidden.reshape(-1, orig_shape[-1])
        input_ids = input_ids.reshape(-1)
        fc = get_forward_context()
        md = fc.attn_metadata
        T = hidden.shape[0]
        hc_hidden = self.H * self.hc
        if md is None or self.kv_cache is None:  # profiling / dummy run: no state, no history
            seqs = [(0, T)]
            ctx_cpu = [torch.full((self.ctx,), self.hasher.eos, dtype=torch.long, device="cpu")]
            conv_hist = None
            load = store = None
            is_prompt = True
            valid = None
        else:
            is_prompt = bool(getattr(md, "is_prompt", False))
            load, store = self._slots(md, "load_indices_tensor"), self._slots(md, "store_indices_tensor")
            store = load if store is None else store
            n = int(load.numel())
            state = self.kv_cache[1].view(self.kv_cache[1].shape[0], -1)
            rows = state.index_select(0, load.long())[:, :self.state_numel]
            has_init = getattr(md, "has_initial_states_p", None) if is_prompt else None
            keep = (load.reshape(-1) != 0)
            if has_init is not None:
                keep = keep & has_init.bool().view(-1)
            rows = torch.where(keep.view(-1, 1), rows, torch.zeros_like(rows))
            conv_hist = rows[:, :self.hist_f32].contiguous().view(torch.bfloat16)[:, :self.state_len * hc_hidden]
            conv_hist = conv_hist.reshape(n, self.state_len, hc_hidden)
            tok_rows = rows[:, self.hist_f32:].cpu().round().long()
            if not bool(keep.all()):
                tok_rows = torch.where(keep.cpu().view(-1, 1), tok_rows, torch.full_like(tok_rows, self.hasher.eos))
            ctx_cpu = list(tok_rows)
            L = T // n
            seqs = [(i * L, L) for i in range(n)]
            pm = getattr(md, "padding_mask_flat", None)
            valid = pm.view(n, L).bool() if (is_prompt and pm is not None and pm.numel() == T) else None

        # ---- n-gram ids on CPU (exact int64), lookup on device
        ids_cpu = input_ids.cpu().long()
        lens = [L for _, L in seqs] if valid is None else valid.sum(-1).cpu().tolist()
        all_ids = torch.zeros(T, self.hasher.num_heads, dtype=torch.long, device="cpu")
        new_ctx = []
        for i, (s, L) in enumerate(seqs):
            nv = int(lens[i])
            toks = ids_cpu[s:s + nv]
            if nv > 0:  # warmup pads with all-padding sequences
                all_ids[s:s + nv] = self.hasher.ids(toks, ctx_cpu[i])
            new_ctx.append(torch.cat([ctx_cpu[i], toks])[-self.ctx:])
        emb = self.ple_embedding.ngram_embedding(all_ids.to(torch.int32).to(hidden.device)).flatten(-2)

        key = self.norm_key(self.key_proj(emb)).unflatten(-1, (self.hc, self.H))
        value = self.value_proj(emb)
        query = self.norm_query(hidden).unflatten(-1, (self.hc, self.H))
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        gated_normed = self.norm_conv(gated)
        if valid is not None:
            vm = valid.reshape(-1, 1).to(gated.dtype)
            gated, gated_normed = gated * vm, gated_normed * vm

        # ---- dilated causal depthwise conv per sequence, with history from the state slot
        n = len(seqs)
        L = seqs[0][1]
        x = gated_normed.view(n, L, hc_hidden)
        hist = conv_hist.to(x.dtype) if conv_hist is not None else x.new_zeros(n, self.state_len, hc_hidden)
        xp = torch.cat([hist, x], dim=1)  # [n, state_len + L, C]
        w = self.conv1d.weight.view(hc_hidden, self.kernel)  # w[:, j] multiplies x[t - (K-1-j)*dil]
        out = x.new_zeros(n, L, hc_hidden)
        for j in range(self.kernel):
            off = j * self.dil
            out = out + xp[:, off:off + L] * w[:, j]
        out = F.silu(out).view(T, hc_hidden)

        # ---- write back state (last state_len valid positions + last ctx token ids)
        if store is not None:
            lens_t = torch.tensor(lens, device=x.device)
            gather = (lens_t.view(-1, 1) + torch.arange(self.state_len, device=x.device).view(1, -1))  # into xp
            new_hist = torch.gather(xp, 1, gather.unsqueeze(-1).expand(-1, -1, hc_hidden))
            new_tok = torch.stack(new_ctx).to(torch.float32).to(x.device)
            hist_bf16 = new_hist.reshape(n, -1).to(torch.bfloat16)
            if hist_bf16.shape[1] % 2:
                hist_bf16 = torch.cat([hist_bf16, hist_bf16.new_zeros(n, 1)], dim=1)
            new_rows = torch.cat([hist_bf16.contiguous().view(torch.float32), new_tok], dim=1)
            st = self.kv_cache[1].view(self.kv_cache[1].shape[0], -1)
            st[:, :self.state_numel].index_copy_(0, store.long(), new_rows.to(st.dtype))
        return (hidden + gated + out).view(orig_shape)


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

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.ple is not None:
            hidden = self.ple(hidden, input_ids)
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
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            HpuQwen4ExpDecoderLayer(vllm_config, config.layer_types[i], prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        for layer in self.layers:
            if layer.ple is not None:
                layer.ple.check_capacity()

    def forward(self, input_ids, positions, inputs_embeds=None) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        h = h.repeat(*([1] * (h.dim() - 1)), self.config.hc_count)  # HPU passes [B, L] or [T] inputs
        for layer in self.layers:
            h = layer(h, positions, input_ids)
        return self.hyper_connection_mixer(h)


class HpuQwen4ExpForConditionalGeneration(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE):
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
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds)

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
