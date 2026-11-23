"""Greedy tokens and margins: alone and batched (run LABEL), or the summary over pairs (compare A:SET B:SET)."""
import json, os, statistics as st, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
S = os.path.dirname(os.path.abspath(__file__))
P = ["The capital of France is", "Explain how a transistor works.", "def quicksort(a):", "Once upon a time",
     "The best programming language is", "List the planets of the solar system:", "La mer est", "1, 1, 2, 3, 5, 8,"]
T = [f"Write a detailed article about the history of {t}.\n" for t in ("the sea", "steel", "bread", "glaciers", "compilers",
     "bees", "railways", "coffee", "volcanoes", "violins", "satellites", "rice", "bridges", "whales", "printing", "salt")]
def one(p):
    body = dict(model="qwen38fn", prompt=p, max_tokens=128, temperature=0, logprobs=5, ignore_eos=True,
                return_token_ids=True, return_tokens_as_token_ids=True)
    req = urllib.request.Request("http://localhost:8000/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1500))["choices"][0]
    top = [{k.split(":")[-1]: v for k, v in d.items()} for d in r["logprobs"]["top_logprobs"]]
    bad = sum(lp < max(t.values()) - 1e-4 for lp, t in zip(r["logprobs"]["token_logprobs"], top))
    return dict(t=r["token_ids"], top=top, bad=bad)
if sys.argv[1] == "run":
    for name, users in (("alone", 1), ("batched", 0), ("batched2", 0)):
        try:
            out = []
            for group in (P, T):
                with ThreadPoolExecutor(users or len(group)) as ex: out += list(ex.map(one, group))
            json.dump(out, open(f"{S}/eq3_{sys.argv[2]}:{name}.json", "w"))
            print(f"eq3 {sys.argv[2]} {name}: {len(out)} outputs, {sum(r['bad'] for r in out)} tokens not top-1", flush=True)
        except Exception as e:
            print(f"eq3 {sys.argv[2]} {name} failed: {e!r}", flush=True)
else:
    try:
        a, b = (json.load(open(f"{S}/eq3_{x}.json")) for x in sys.argv[2:4])
    except OSError as e:
        sys.exit(f"compare: {e}")
    same, flips, d = 0, [], []
    for x, y in zip(a, b):
        n = next((j for j, (u, v) in enumerate(zip(x["t"], y["t"])) if u != v), None)
        same += n is None
        if n is not None:
            t = x["top"][n]
            flips.append(t.get(str(x["t"][n]), -99) - t.get(str(y["t"][n]), -99))
        for s, t in list(zip(x["top"], y["top"]))[: None if n is None else n + 1]:
            k, l = sorted(s, key=s.get, reverse=True)[:2]
            if k in t and l in t: d.append(abs((s[k] - s[l]) - (t[k] - t[l])))
    d.sort()
    print(f"{sys.argv[2]:16s} vs {sys.argv[3]:16s} identical {same:2d}/{len(a)}, largest flip {max(flips, default=0):.3f}, margin drift "
          f"n={len(d)} median {st.median(d):.3f} p99 {d[int(.99 * len(d))]:.3f} max {d[-1]:.3f}")
