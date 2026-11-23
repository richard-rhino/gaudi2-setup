# SPDX-License-Identifier: Apache-2.0
"""CPU check: attn_static's bucketed decode must equal the reference attention exactly, on a small random model.

top-k is given a position-ordered tie-break for both sides: torch's own order among equal scores depends on the
tensor width, and the indexer's relu produces exact ties.

PYTHONPATH=<snap>/inference python test_attn_static.py
"""
import os

os.environ["DSV41_ATTN_BUCKET_MIN"] = "64"
import torch

import generate_hpu as G
import attn_static

CFG = dict(
    max_batch_size=1, max_seq_len=256, vocab_size=2000, dim=256, moe_inter_dim=128, n_layers=5, n_mtp_layers=0,
    n_heads=8, n_routed_experts=4, q_lora_rank=64, head_dim=64, rope_head_dim=16, o_groups=2, o_lora_rank=64,
    window_size=32, compress_ratios=(0, 2, 2, 1, 1), kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
    index_n_heads=4, index_head_dim=32, index_topk=16, candidate_source_layer=3, candidate_topk_blocks=8,
    candidate_block_size=4,
)


def fill(model, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.dtype == torch.uint8:
            lo, hi = (120, 124) if name.endswith("scale") else (0, 256)
            p.data.copy_(torch.randint(lo, hi, p.shape, generator=g, dtype=torch.int32).to(torch.uint8))
        else:
            p.data.copy_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))


@torch.inference_mode()
def run(model, toks, prompt):
    out = [model(toks[:, :prompt], 0)[1].float().clone()]
    for i in range(prompt, toks.size(1)):
        out.append(model(toks[:, i:i + 1], i)[1].float().clone())
    return out


def stable_topk(self, k, dim=-1, largest=True, sorted=True):
    r = torch.sort(self, dim=dim, descending=largest, stable=True)
    return torch.return_types.topk((r.values.narrow(dim, 0, k), r.indices.narrow(dim, 0, k)))


def main():
    torch.set_num_threads(16)
    torch.Tensor.topk = stable_topk
    worst, same, n = 0.0, 0, 0
    for seed, prompt in ((0, 1), (1, 7), (2, 37), (3, 100)):
        model, _ = G.build(dict(CFG), "cpu")
        fill(model, seed)
        toks = torch.randint(0, 2000, (1, 200), generator=torch.Generator().manual_seed(seed + 10))
        attn_static.uninstall()
        ref = run(model, toks, prompt)
        attn_static.install(compile_pieces=False)
        new = run(model, toks, prompt)
        for a, b in zip(ref, new):
            worst = max(worst, ((a - b).norm() / a.norm()).item())
            same += int(torch.equal(a.argmax(-1), b.argmax(-1)))
            n += 1
        print(f"seed {seed} prompt {prompt}: worst rel err so far {worst:.3g}, argmax equal {same}/{n}", flush=True)
    assert worst == 0.0 and same == n, "static attention differs from the reference"


if __name__ == "__main__":
    main()
