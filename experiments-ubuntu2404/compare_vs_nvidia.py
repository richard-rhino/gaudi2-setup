#!/usr/bin/env python3
"""Compare a Gaudi vLLM server with the Nvidia server for the same model (no torch needed).
1) teacher-forced: logprob of every prompt token (prompt_logprobs) -> mean/max |difference|
2) greedy: first 64 generated tokens, length of the identical prefix.
Usage: compare_vs_nvidia.py [gaudi_url model] [ref_url model]"""
import json, sys, urllib.request
GAUDI = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("http://localhost:8000", "qwen38fn")
REF = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("http://10.40.0.248:8000", "local")
PROMPTS = [
    "The capital of France is Paris. The capital of Japan is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain in two sentences why the sky is blue.",
    ("In 1969, Apollo 11 landed on the Moon. Neil Armstrong and Buzz Aldrin walked on the surface while Michael "
     "Collins orbited above. The mission returned safely on July 24. ") * 20 + "Summarize the mission in one line:",
]

def post(base, body):
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def run(srv, prompt):
    tf = post(srv[0], {"model": srv[1], "prompt": prompt, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    pl = tf["choices"][0]["prompt_logprobs"]
    gen = post(srv[0], {"model": srv[1], "prompt": prompt, "max_tokens": 64, "temperature": 0, "logprobs": 0})
    return pl, gen["choices"][0]["text"]

for p in PROMPTS:
    gpl, gtxt = run(GAUDI, p)
    rpl, rtxt = run(REF, p)
    d = []
    for a, b in zip(gpl, rpl):
        if a is None or b is None:
            continue
        # each entry: {token_id: {"logprob":..}} containing at least the actual prompt token
        common = set(a) & set(b)
        for t in common:
            d.append(abs(a[t]["logprob"] - b[t]["logprob"]))
    same = next((i for i, (x, y) in enumerate(zip(gtxt, rtxt)) if x != y), min(len(gtxt), len(rtxt)))
    print(f"prompt {len(p):5d} chars: prompt-token |dlogprob| mean={sum(d)/max(len(d),1):.4f} max={max(d, default=0):.3f} "
          f"(n={len(d)}); greedy identical for {same}/{max(len(gtxt), len(rtxt))} chars")
    print("   gaudi:", repr(gtxt[:120])); print("   nvidia:", repr(rtxt[:120]))
