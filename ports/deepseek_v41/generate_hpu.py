"""Run DeepSeek-V4.1's reference implementation on Gaudi (HPU), one process per card.

  # real model (after `python convert.py --hf-ckpt-path <snap> --save-path /models/dsv41-mp8 --mp 8 --expert-dtype fp4`):
  torchrun --nproc-per-node 8 generate_hpu.py --ckpt-path /models/dsv41-mp8 --config <snap>/inference/config.json \
           --prompts prompts.txt --max-new-tokens 128
  # tiny random self-test, CPU vs HPU (single process):
  python generate_hpu.py --selftest
"""
import argparse, json, os, sys, time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpu_patches  # noqa: E402

hpu_patches.install()
import model as M  # noqa: E402


def setup(device: str):
    ws, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    if device == "hpu":
        import habana_frameworks.torch.core  # noqa: F401
        import habana_frameworks.torch.distributed.hccl  # noqa: F401
    if ws > 1:
        dist.init_process_group("hccl" if device == "hpu" else "gloo")
    return ws, rank


def build(args_dict: dict, device: str, tokenizer=None):
    torch.set_default_dtype(torch.bfloat16)
    args = M.ModelArgs(**args_dict)
    with torch.device(device):
        model = M.Transformer(args, tokenizer) if tokenizer is not None else M.Transformer(args)
    hpu_patches.move_hash_to_cpu(model)
    return model, args


@torch.inference_mode()
def greedy(model, prompt_tokens: list[list[int]], max_new: int, eos: int, device: str):
    """Batch-1-at-a-time greedy decode with the reference's (start_pos) KV-cache API; returns tokens and timings."""
    outs, stats = [], []
    for toks in prompt_tokens:
        x = torch.tensor([toks], device=device)
        t0 = time.time()
        ids, logits, _ = model(x, 0)
        if device == "hpu":
            torch.hpu.synchronize()
        t_prefill = time.time() - t0
        nxt = int(logits[0].argmax()) if logits.dim() == 2 else int(logits[0, -1].argmax())
        gen = [nxt]
        t1 = time.time()
        for i in range(max_new - 1):
            ids, logits, _ = model(torch.tensor([[nxt]], device=device), len(toks) + i)
            nxt = int(logits[0].argmax()) if logits.dim() == 2 else int(logits[0, -1].argmax())
            gen.append(nxt)
            if nxt == eos:
                break
        t_dec = time.time() - t1
        outs.append(gen)
        stats.append({"prompt": len(toks), "prefill_s": t_prefill, "decode_tok_s": (len(gen) - 1) / max(t_dec, 1e-9)})
    return outs, stats


def selftest():
    """Tiny random V4.1-shaped model (reference defaults + engram off): same weights on CPU and HPU, compare logits."""
    torch.manual_seed(0)
    cfg = dict(dspark_block_size=6, dspark_target_layer_ids=(3, 4), max_batch_size=1, max_seq_len=512)
    res = {}
    for dev in ("cpu", "hpu"):
        setup(dev)
        model, args = build(cfg, dev)
        if dev == "cpu":
            g = torch.Generator().manual_seed(1)
            state = {}
            for k, p in model.state_dict().items():
                if p.dtype == torch.uint8:
                    hi = 256 if not k.endswith(".scale") else 4
                    v = torch.randint(0, hi, p.shape, generator=g, dtype=torch.int32).to(torch.uint8)
                    if k.endswith(".scale"):
                        v = v + 120  # E8M0 exponents around 2^-7..2^-4
                    state[k] = v
                elif p.is_floating_point():
                    state[k] = (torch.randn(p.shape, generator=g) * 0.02).to(p.dtype)
                else:
                    state[k] = p
        model.load_state_dict(state, strict=False)
        x = torch.randint(0, args.vocab_size, (1, 150), generator=torch.Generator().manual_seed(2))
        _, logits, _ = model(x[:, :128].to(dev), 0)
        steps = [logits.float().cpu()]
        for i in range(128, 140):
            _, logits, _ = model(x[:, i:i + 1].to(dev), i)
            steps.append(logits.float().cpu())
        res[dev] = steps
    for i, (a, b) in enumerate(zip(res["cpu"], res["hpu"])):
        rel = ((a - b).norm() / a.norm()).item()
        print(f"step {i:2d} rel_err={rel:.4f} argmax_equal={bool((a.argmax(-1) == b.argmax(-1)).all())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt-path"); ap.add_argument("--config"); ap.add_argument("--prompts")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ws, rank = setup("hpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt_path)
    cfg = json.load(open(a.config)); cfg.update(max_batch_size=1, max_seq_len=8192)
    t0 = time.time()
    model, args = build(cfg, "hpu", tok)
    hpu_patches.load_checkpoint(model, os.path.join(a.ckpt_path, f"model{rank}-mp{ws}.safetensors"))
    if rank == 0:
        print(f"model built + loaded in {time.time() - t0:.0f}s", flush=True)
    from encoding import encode_messages
    prompts = open(a.prompts).read().strip().split("\n\n")
    toks = [tok.encode(encode_messages([{"role": "user", "content": p}], thinking_mode="chat")) for p in prompts]
    outs, stats = greedy(model, toks, a.max_new_tokens, tok.eos_token_id, "hpu")
    if rank == 0:
        for p, o, s in zip(prompts, outs, stats):
            print(json.dumps({"prompt": p[:80], **s}), "\n", tok.decode(o), "\n", flush=True)
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
