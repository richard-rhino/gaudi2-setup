"""Teacher-forced check: per prompt position, vLLM top-1 (prompt_logprobs) vs HF argmax. Also single vs batched."""
import os, sys, json, torch

def main():
    path = sys.argv[1]; tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    torch.manual_seed(1)
    lens = [int(x) for x in os.environ.get("LENS", "7,33,130,300").split(",")]
    prompts = [torch.randint(10, 240000, (L,)).tolist() for L in lens]
    if os.environ.get("STAGE") == "hf":
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).eval()
        out = []
        for p in prompts:
            with torch.no_grad():
                lg = m(input_ids=torch.tensor([p])).logits[0].float()
            out.append({"argmax": lg.argmax(-1).tolist(), "lse": torch.logsumexp(lg, -1).tolist(),
                        "tok_logit": lg[torch.arange(len(p) - 1), torch.tensor(p[1:])].tolist()})
        json.dump(out, open(path + "/hf_tf.json", "w")); print("hf done"); return
    from vllm import LLM, SamplingParams
    llm = LLM(model=path, tensor_parallel_size=tp, max_model_len=2048, max_num_seqs=int(os.environ.get("SEQS", 8)),
              dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.5,
              limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=False)
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    ref = json.load(open(path + "/hf_tf.json"))
    res = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    for L, p, r, h in zip(lens, prompts, res, ref):
        pl = r.prompt_logprobs  # list (None for pos 0) of {tok: Logprob}
        top1 = []
        lp_diff = []
        for i in range(1, L):
            d = pl[i]
            best = min(d.items(), key=lambda kv: kv[1].rank)[0]
            top1.append(best)
            hf_lp = h["tok_logit"][i - 1] - h["lse"][i - 1]
            lp_diff.append(abs(d[p[i]].logprob - hf_lp))
        agree = sum(a == b for a, b in zip(top1, h["argmax"][:L - 1]))
        first_bad = next((i for i, (a, b) in enumerate(zip(top1, h["argmax"][:L - 1])) if a != b), None)
        print(f"len={L:4d} top1 agree={agree}/{L-1}  first_mismatch_pos={first_bad}  "
              f"mean|dlogprob|={sum(lp_diff)/len(lp_diff):.3f} max={max(lp_diff):.3f}")

if __name__ == "__main__":
    main()
