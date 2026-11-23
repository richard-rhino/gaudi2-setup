# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next MTP draft model (``Qwen4ExpMTP``) for Intel Gaudi, text only.

One dense-attention decoder layer of the HPU Qwen4Exp port (no GDN, no PLE). It starts from the pre-mixer hidden
streams of the target, or of the previous draft step, and returns ``(sample [.., H], streams [.., hc * H])``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_gaudi.models.qwen4_exp import GatedResidual, GroupedGemmaRMSNorm, HpuQwen4ExpDecoderLayer


def remap_mtp_weight_name(name: str) -> str | None:
    """Checkpoint name -> draft model name, or None for tensors the draft model does not use."""
    name = name.removeprefix("model.language_model.")
    if name.startswith("embed_tokens."):
        return f"model.{name}"
    if name.startswith("lm_head."):
        return name
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    return None


class HpuQwen4ExpPredictor(nn.Module):

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if (getattr(config, "mtp", None) or {}).get("num_hidden_layers", 1) != 1:
            raise NotImplementedError("Qwen4Exp MTP on HPU supports one MTP layer.")
        self.H, self.hc = config.hidden_size, config.hc_count
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, self.H)
        self.pre_fc_norm_embedding = GroupedGemmaRMSNorm(self.H, None, config.rms_norm_eps)
        self.pre_fc_norm_hidden = GroupedGemmaRMSNorm(self.hc * self.H, None, config.rms_norm_eps)
        self.fc_embedding = nn.Linear(self.H, self.H, bias=False)
        self.fc_hidden = nn.Linear(self.H, self.H, bias=False)
        # Numbered after the target's layers, so that KV cache and MoE layer ids do not collide. No PLE: that index
        # is not a target layer, even when it appears in ple_layer_ids (configs with fewer layers, as in tests).
        self.layers = nn.ModuleList([
            HpuQwen4ExpDecoderLayer(vllm_config,
                                    "full_attention",
                                    prefix=f"mtp.layers.{config.num_hidden_layers}",
                                    use_ple=False)
        ])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    # Compiled as regions of their own by the HPU runner (regional compilation); eager, each of their ops would be a
    # separate graph launch for every draft token.
    hpu_compile_regions = ("input_streams", "hyper_connection_mixer")

    def input_streams(self, embeds: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Token embedding + the target's hidden streams -> the MTP layer's input streams."""
        embeds = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        streams = hidden_states.reshape(*embeds.shape[:-1], self.hc * self.H)  # [T] or [B, L] tokens
        streams = self.pre_fc_norm_hidden(streams).unflatten(-1, (self.hc, self.H))
        # The token is a residual of unit weight on every stream
        return (self.fc_hidden(streams) + embeds.unsqueeze(-2)).flatten(-2)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        streams = self.input_streams(embeds, hidden_states)
        streams = self.layers[0](streams, positions, None)
        return self.hyper_connection_mixer(streams), streams


class HpuQwen4ExpMTP(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.model = HpuQwen4ExpPredictor(vllm_config)
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, hidden_states=None, inputs_embeds=None, **kwargs):
        assert hidden_states is not None, "the MTP draft model starts from hidden streams"
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = Qwen3_5Model.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={
                ".self_attn.indexer.": None,  # dense attention, like the target
                "hyper_connection_mixer.block_inject_weight": None,
            })
        renamed = ((new, w) for name, w in weights if (new := remap_mtp_weight_name(name)) is not None)
        return AutoWeightsLoader(self).load_weights(renamed, mapper=mapper)
