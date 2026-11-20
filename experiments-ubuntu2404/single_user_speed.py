#!/usr/bin/env python3
"""Light single-user decode speed of an OpenAI-compatible server: 3 streamed requests, 256 tokens each.
Usage: single_user_speed.py <base_url> <model>"""
import json, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
for i in range(3):
    body = {"model": model, "prompt": f"Write a long story about a lighthouse keeper, part {i}.", "max_tokens": 256,
            "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; toks = None
    for line in urllib.request.urlopen(req, timeout=600):
        if line.startswith(b"data: {"):
            d = json.loads(line[6:])
            if d.get("choices") and first is None: first = time.time()
            if d.get("choices"): n += 1
            if d.get("usage"): toks = d["usage"]["completion_tokens"]
    t1 = time.time()
    print(f"run {i}: TTFT {first - t0:.2f} s, {toks} tokens in {n} chunks ({toks / n:.2f} tok/chunk), "
          f"decode {(toks - 1) / (t1 - first):.1f} tok/s")
