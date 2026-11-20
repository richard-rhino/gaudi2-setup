"""Run the HF reference Qwen4Exp on CPU and HPU (eager), compare prefill logits + greedy decode."""
import sys, time, torch
import habana_frameworks.torch.core as htcore  # noqa
sys.path.insert(0, "/home/ubun/work/qwen4"); import hpu_compat; hpu_compat.install_all(); hpu_compat.patch_hf_ngram_cpu(); hpu_compat.install_scatter_bool()
from transformers import AutoModelForImageTextToText, AutoConfig
path = sys.argv[1]
torch.manual_seed(0)
ids = torch.randint(0, 248000, (1, 48))
res = {}
for dev in ["cpu", "hpu"]:
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    t = time.time()
    with torch.no_grad():
        logits = m(input_ids=ids.to(dev)).logits.float().cpu()
        gen = m.generate(input_ids=ids.to(dev), max_new_tokens=16, do_sample=False).cpu()
    res[dev] = (logits, gen[0, ids.shape[1]:])
    print(dev, "time %.1fs" % (time.time() - t), "gen", gen[0, ids.shape[1]:].tolist())
a, b = res["cpu"][0], res["hpu"][0]
print("prefill logits max abs diff %.4f, rel %.4f, argmax agree %.3f" % (
    (a - b).abs().max(), ((a - b).norm() / a.norm()), (a.argmax(-1) == b.argmax(-1)).float().mean()))
print("greedy tokens agree:", (res["cpu"][1] == res["hpu"][1]).float().mean().item())
