"""Decode-path check: vLLM generates N tokens (logprobs from the decode steps); HF scores the same sequence."""
import os, sys, json, torch

def main():
    path = sys.argv[1]; tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    torch.manual_seed(1)
    lens = [int(x) for x in os.environ.get("LENS", "7,33,130,300").split(",")]
    prompts = [torch.randint(10, 240000, (L,)).tolist() for L in lens]
    if os.environ.get("STAGE") == "hf":
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).eval()
        gen = json.load(open(path + "/vllm_gen.json"))
        for L, p, g in zip(lens, prompts, gen):
            seq = p + g["toks"]
            with torch.no_grad():
                lg = m(input_ids=torch.tensor([seq])).logits[0].float()
            lp = torch.log_softmax(lg, -1)
            hf = [lp[L - 1 + i, t].item() for i, t in enumerate(g["toks"])]
            d = [abs(a - b) for a, b in zip(hf, g["lps"])]
            print(f"len={L:4d} decode steps={len(d)} mean|dlogprob|={sum(d)/len(d):.3f} max={max(d):.3f} "
                  f"last8_mean={sum(d[-8:])/8:.3f}")
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=path, tensor_parallel_size=tp, max_model_len=2048, max_num_seqs=8, dtype="bfloat16",
              enforce_eager=os.environ.get("EAGER", "1") == "1", gpu_memory_utilization=0.5, limit_mm_per_prompt={"image": 0, "video": 0},
              enable_prefix_caching=False)
    sp = SamplingParams(temperature=0, max_tokens=int(os.environ.get("N_NEW", 32)), ignore_eos=True, logprobs=0)
    res = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    out = []
    for r in res:
        o = r.outputs[0]
        out.append({"toks": list(o.token_ids), "lps": [lp[t].logprob for lp, t in zip(o.logprobs, o.token_ids)]})
    json.dump(out, open(path + "/vllm_gen.json", "w")); print("vllm done")

if __name__ == "__main__":
    main()
