"""Per-module synchronized wall time of DeepSeek-V4.1 decode steps on HPU (rank 0 prints a table).
torchrun --nproc-per-node 4 time_modules.py --ckpt-path ... --config ..."""
import argparse, json, os, sys, time, collections
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_hpu as G
import hpu_patches
M = G.M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt-path"); ap.add_argument("--config"); ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    ws, rank = G.setup("hpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt_path)
    cfg = json.load(open(a.config)); cfg.update(max_batch_size=1, max_seq_len=1024)
    model, args = G.build(cfg, "hpu", tok)
    hpu_patches.load_checkpoint(model, a.ckpt_path, rank, ws)
    if os.getenv("DSV41_MOE_STACK", "1") == "1":
        hpu_patches.stack_experts(model); hpu_patches.install_moe()
    if os.getenv("DSV41_COMPILE", "0") == "1":
        hpu_patches.install_compile()
    from encoding import encode_messages
    ids = tok.encode(encode_messages([{"role": "user", "content": "Tell me about Tokyo."}], thinking_mode="chat"), add_special_tokens=False)
    acc = collections.defaultdict(float); on = [False]; t0s = {}
    def pre(name):
        def f(mod, inp):
            if on[0]:
                torch.hpu.synchronize(); t0s[name] = time.perf_counter()
        return f
    def post(name):
        def f(mod, inp, out):
            if on[0]:
                torch.hpu.synchronize(); acc[name] += time.perf_counter() - t0s[name]
        return f
    for n, m in model.named_modules():
        sub = os.getenv("HOOK_SUB")  # e.g. "attn": time layers.N.attn.<child> instead
        if n.startswith("layers.") and (n.count(".") == 2 if not sub else (n.count(".") == 3 and n.split(".")[2] == sub)):
            key = n.split(".", 2)[2]
            m.register_forward_pre_hook(pre(n)); m.register_forward_hook(post(n))
        if n in ("embed", "head", "norm"):
            m.register_forward_pre_hook(pre(n)); m.register_forward_hook(post(n))
    with torch.device("hpu"), torch.inference_mode():
        _, logits, _ = model(torch.tensor([ids], device="hpu"), 0)
        nxt = int(logits[0].argmax()) if logits.dim() == 2 else int(logits[0, -1].argmax())
        for i in range(int(os.getenv('WARM_STEPS', '2'))):  # warm the decode shapes
            _, logits, _ = model(torch.tensor([[nxt]], device="hpu"), len(ids) + i); nxt = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        torch.hpu.synchronize(); on[0] = True; t = time.perf_counter()
        for i in range(a.steps):
            _, logits, _ = model(torch.tensor([[nxt]], device="hpu"), len(ids) + 2 + i); nxt = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        torch.hpu.synchronize(); total = (time.perf_counter() - t) / a.steps
    if rank == 0:
        by = collections.defaultdict(float)
        for n, v in acc.items():
            by[n.split(".", 2)[2] if n.startswith("layers.") else n] += v / a.steps
        print(f"decode step {total*1e3:.0f} ms (with per-module syncs); per kind (sum over layers):", flush=True)
        for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"  {k:20s} {v*1e3:8.1f} ms", flush=True)
        print(f"  {'(not in hooks)':20s} {(total - sum(by.values()))*1e3:8.1f} ms", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
