"""Load the converted checkpoint like generate_hpu.py, run one prefill, report the first modules whose output is
non-finite or huge (forward hooks, rank 0 prints). torchrun --nproc-per-node 4 debug_nan.py --ckpt-path ... --config ..."""
import argparse, json, os, sys, time
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_hpu as G  # installs the patches
import hpu_patches
M = G.M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt-path"); ap.add_argument("--config")
    a = ap.parse_args()
    ws, rank = G.setup("hpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt_path)
    cfg = json.load(open(a.config)); cfg.update(max_batch_size=1, max_seq_len=1024)
    model, args = G.build(cfg, "hpu", tok)
    hpu_patches.load_checkpoint(model, a.ckpt_path, rank, ws)
    # parameter sanity: non-finite values in any float parameter
    if rank == 0:
        bad = [n for n, p in model.named_parameters() if p.is_floating_point() and not torch.isfinite(p).all().item()]
        print("non-finite params:", bad[:10], flush=True)
    log = []
    def hook(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                tf = t.float()
                log.append((name, bool(torch.isfinite(tf).all().item()), float(tf.abs().max().item()) if tf.numel() else 0.0,
                            tuple(t.shape)))
        return f
    for n, m in model.named_modules():
        if n and n.count(".") <= 3 and not n.startswith("vision"):
            m.register_forward_hook(hook(n))
    from encoding import encode_messages
    ids = tok.encode(encode_messages([{"role": "user", "content": "What is the capital of Japan?"}], thinking_mode="chat"),
                     add_special_tokens=False)
    with torch.device("hpu"), torch.inference_mode():
        _, logits, _ = model(torch.tensor([ids], device="hpu"), 0)
    torch.hpu.synchronize()
    if rank == 0:
        print("prompt ids:", ids, flush=True)
        first_bad = next((i for i, r in enumerate(log) if not r[1] or r[2] > 1e4), None)
        lo = max(0, (first_bad or 0) - 8)
        for r in log[lo:lo + 30]:
            print(f"{r[0]:45s} finite={r[1]} maxabs={r[2]:.4g} {r[3]}", flush=True)
        print("logits finite:", bool(torch.isfinite(logits.float()).all()), "top5:", logits.float().topk(5).indices.tolist(), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
