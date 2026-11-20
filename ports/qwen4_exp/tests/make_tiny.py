"""Build a tiny random-weight Qwen4Exp checkpoint with the real model's architecture features."""
import json, sys, torch
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
real = json.load(open(sys.argv[1]))
out = sys.argv[2]
tc = dict(real["text_config"])
tc.update(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=256,
          linear_num_key_heads=4, linear_num_value_heads=16, linear_key_head_dim=32, linear_value_head_dim=32,
          num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64, shared_expert_intermediate_size=64,
          hc_lowrank=32, ple_embed_dim=256, ngram_vocab_size_base=40000, split_ngram_parts=4,
          indexer_budget=2048, indexer_head_dim=128, layer_types=None, full_attention_interval=4,
          mtp_num_hidden_layers=0)
tc.pop("layer_types", None); tc["full_attention_interval"] = 4
tc["mtp"] = None
vc = dict(real["vision_config"]); vc.update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256)
cfg = dict(real); cfg["text_config"] = tc; cfg["vision_config"] = vc
cfg = Qwen4ExpConfig(**cfg)
torch.manual_seed(0)
m = Qwen4ExpForConditionalGeneration(cfg).to(torch.bfloat16)
print("params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
m.save_pretrained(out)
