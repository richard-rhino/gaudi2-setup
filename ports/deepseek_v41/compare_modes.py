# SPDX-License-Identifier: Apache-2.0
"""Real model on HPU: does each speed-up change the output, and how fast is it?

1. reference:  per-expert loop, eager attention (the slow path); greedy, its tokens are forced on the other modes.
2. stacked:    stacked-expert gather MoE, still eager.
3. compiled:   stacked + install_compile (with the static-shape attention); run twice (cold, warm).
Last, a free greedy run in the compiled mode, for text and tok/s.
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

import generate_hpu as G
import hpu_patches


@torch.inference_mode()
def run(model, ids, n, eos, forced=None):
    lps, toks, dts = [], [], []
    x, pos = torch.tensor([ids], device="hpu"), 0
    for i in range(n):
        t0 = time.perf_counter()
        with torch.device("hpu"):
            logits = model(x, pos)[1]
        lp = logits.reshape(-1, logits.shape[-1])[-1].float().cpu()
        dts.append(time.perf_counter() - t0)
        lp = torch.log_softmax(lp, -1)
        tok = int(lp.argmax()) if forced is None else forced[i]
        lps.append(lp)
        toks.append(tok)
        if (forced is None and tok == eos) or (forced is not None and i + 1 == len(forced)):
            break
        pos = pos + x.size(1)
        x = torch.tensor([[tok]], device="hpu")
    return toks, lps, dts


def report(name, base, got):
    d = [abs(float(g[int(b.argmax())] - b.max())) for b, g in zip(base, got)]
    diff = [i for i, (b, g) in enumerate(zip(base, got)) if int(b.argmax()) != int(g.argmax())]
    margins = []
    for i in diff:
        top = base[i].topk(2).values
        margins.append(round(float(top[0] - top[1]), 3))
    return dict(mode=name, steps=len(d), dlp_mean=round(statistics.mean(d), 4), dlp_max=round(max(d), 4),
                top1_differs_at=diff, reference_margin_there=margins)


def speed(dts):
    d = sorted(dts[1:]) or [float("nan")]
    return dict(prefill_s=round(dts[0], 2), decode_ms_p50=round(d[len(d) // 2] * 1e3, 1),
                decode_ms_max=round(d[-1] * 1e3, 1), decode_tok_s=round(len(d) / sum(d), 3))


def main():
    ap = argparse.ArgumentParser()
    for a in ("--ckpt-path", "--config", "--prompts"):
        ap.add_argument(a)
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()
    ws, rank = G.setup("hpu")
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt_path)
    cfg = json.load(open(a.config))
    cfg.update(max_batch_size=1, max_seq_len=8192)
    model, _ = G.build(cfg, "hpu", tok)
    hpu_patches.load_checkpoint(model, a.ckpt_path, rank, ws)
    log("loaded")
    from encoding import encode_messages
    prompts = open(a.prompts).read().strip().split("\n\n")
    ids = [tok.encode(encode_messages([{"role": "user", "content": p}], thinking_mode="chat"),
                      add_special_tokens=False) for p in prompts]

    base = []
    for i, x in enumerate(ids):
        toks, lps, dts = run(model, x, a.steps, tok.eos_token_id)
        base.append((toks, lps))
        log(json.dumps(dict(mode="reference", prompt=i, **speed(dts))), repr(tok.decode(toks)))

    def forced(name):
        for i, x in enumerate(ids):
            _, lps, dts = run(model, x, a.steps, tok.eos_token_id, base[i][0])
            log(json.dumps(dict(prompt=i, **report(name, base[i][1], lps), **speed(dts))))

    hpu_patches.stack_experts(model)
    hpu_patches.install_moe()
    forced("stacked")
    hpu_patches.install_compile()
    forced("compiled-cold")
    forced("compiled-warm")
    for i, x in enumerate(ids):
        toks, _, dts = run(model, x, 48, tok.eos_token_id)
        log(json.dumps(dict(mode="compiled-free", prompt=i, **speed(dts))), repr(tok.decode(toks)))
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
