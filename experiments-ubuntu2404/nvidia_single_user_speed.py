"""One streamed request at a time to each LAN Nvidia server, 256 tokens: decode tok/s and TTFT. Light use only: never load-test these hosts."""
import json, time, urllib.request
for name, url, model in (("deb-ai", "http://10.40.0.248:8000/v1/completions", "local"),
                         ("rack-n03", "http://10.40.3.193:8300/v1/completions", "deepseek-v4.1-flash")):
    for run in range(2):
        body = dict(model=model, prompt="Write a long story about a lighthouse keeper.\n", max_tokens=256, ignore_eos=True,
                    temperature=0, stream=True)
        t0, t = time.time(), []
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        for l in urllib.request.urlopen(req, timeout=300):
            if l.startswith(b"data: {") and b'"text"' in l:
                t.append(time.time())
        print(f"{name} run {run}: TTFT {t[0]-t0:.2f} s, {len(t)} chunks, decode {255/(t[-1]-t[0]):.1f} tok/s (if 1 token per chunk)", flush=True)
