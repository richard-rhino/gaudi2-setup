# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced check against another server's greedy output (rack-n03, vLLM on Nvidia).

Feeds the server's prompt and greedy tokens to the Gaudi model and reports, per step, whether Gaudi's top-1 is the
server's token, Gaudi's margin where it is not, and the server's own margin there (from its top-5 logprobs).
Modes: stacked (eager) and compiled (install_compile, with the static-shape attention unless DSV41_ATTN_STATIC=0).
torchrun --nproc-per-node 4 forced_vs_rack.py --ckpt-path ... --config ... --ref ref.json
"""
import argparse
import json

import torch
import torch.distributed as dist

import compare_modes as C
import generate_hpu as G
import hpu_patches


def summary(name, ref, lps):
    rows, flips = [], []
    for j, (want, lp) in enumerate(zip(ref["t"], lps)):
        top2 = lp.topk(2)
        got = int(top2.indices[0])
        rows.append(got == want)
        if got != want:
            srv = sorted(ref["top"][j].values(), reverse=True)
            flips.append(dict(step=j, gaudi_margin=round(float(lp[got] - lp[want]), 3),
                              server_margin=round(srv[0] - srv[1], 3) if len(srv) > 1 else None))
    return dict(mode=name, steps=len(rows), top1_agree=sum(rows), flips=flips)


def main():
    ap = argparse.ArgumentParser()
    for a in ("--ckpt-path", "--config", "--ref"):
        ap.add_argument(a)
    a = ap.parse_args()
    ws, rank = G.setup("hpu")
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt_path)
    cfg = json.load(open(a.config))
    cfg.update(max_batch_size=1, max_seq_len=8192)
    model, _ = G.build(cfg, "hpu", tok)
    hpu_patches.load_checkpoint(model, a.ckpt_path, rank, ws)
    refs = json.load(open(a.ref))
    ids = []
    for r in refs:
        mine = tok.encode(r["text"], add_special_tokens=False)
        same = mine == r["prompt_token_ids"]
        log(f"prompt tokens equal to the server's: {same} ({len(mine)} vs {len(r['prompt_token_ids'])})")
        ids.append(r["prompt_token_ids"])
    hpu_patches.stack_experts(model)
    hpu_patches.install_moe()
    for name in ("stacked", "compiled-cold", "compiled"):
        if name == "compiled-cold":
            hpu_patches.install_compile()
        for i, (x, r) in enumerate(zip(ids, refs)):
            _, lps, _ = C.run(model, x, len(r["t"]), tok.eos_token_id, r["t"])
            if name != "compiled-cold":
                log(json.dumps(dict(prompt=i, **summary(name, r, lps))))
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
