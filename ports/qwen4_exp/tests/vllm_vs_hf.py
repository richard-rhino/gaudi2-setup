"""Greedy-decode the same token-id prompts with vLLM (HPU port) and HF (CPU reference); report agreement."""
import os, sys, json, torch
def main():
    path = sys.argv[1]; tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    N_NEW = int(os.environ.get("N_NEW", 24))
    torch.manual_seed(1)
    lens = [7, 33, 130, 300]
    prompts = [torch.randint(10, 240000, (L,)).tolist() for L in lens]
    if os.environ.get("STAGE", "vllm") == "hf":
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).eval()
        outs = []
        for p in prompts:
            with torch.no_grad():
                g = m.generate(input_ids=torch.tensor([p]), max_new_tokens=N_NEW, do_sample=False, eos_token_id=-1)
            outs.append(g[0, len(p):].tolist())
        json.dump(outs, open(path + "/hf_ref.json", "w")); print("hf done")
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=path, tensor_parallel_size=tp, max_model_len=2048, max_num_seqs=8, dtype="bfloat16",
                  enforce_eager=os.environ.get("EAGER", "1") == "1", gpu_memory_utilization=0.5,
                  limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=False)
        sp = SamplingParams(temperature=0, max_tokens=N_NEW, ignore_eos=True)
        res = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
        got = [list(r.outputs[0].token_ids) for r in res]
        ref = json.load(open(path + "/hf_ref.json"))
        for L, g, r in zip(lens, got, ref):
            first_div = next((i for i, (a, b) in enumerate(zip(g, r)) if a != b), len(r))
            print(f"len={L:4d}  match_prefix={first_div}/{len(r)}  agree={sum(a == b for a, b in zip(g, r))}/{len(r)}")


if __name__ == '__main__':
    main()
