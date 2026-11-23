"""For compileGraph events in the last decode steps, the innermost project Python frames active at (or just before) them."""
import bisect, collections, json, re, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
comp = sorted((e for e in X if e["name"] == "compileGraph"), key=lambda e: e["ts"])
main_tid = collections.Counter(e["tid"] for e in X if e.get("cat") == "python_function").most_common(1)[0][0]
py = sorted((e for e in X if e.get("cat") == "python_function" and e["tid"] == main_tid), key=lambda e: e["ts"])
starts = [e["ts"] for e in py]
keep = re.compile(r"(vllm_gaudi|vllm/v1|vllm/model_executor|qwen4|habana_frameworks/torch/(core|hpu))")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
sites = collections.Counter()
for c in comp[-n:]:
    i = bisect.bisect_right(starts, c["ts"])
    # enclosing frames: scan back for events whose interval covers c; if none cover (background thread), use the last ended
    frames = [e for e in py[max(0, i - 4000):i] if e["ts"] + e["dur"] >= c["ts"]]
    if not frames:
        frames = py[max(0, i - 60):i]
    fr = [e["name"] for e in frames if keep.search(e["name"])]
    sites[" <- ".join(reversed([re.sub(r".*/(src|site-packages)/", "", f)[:95] for f in fr[-3:]]))] += 1
for k, v in sites.most_common(14):
    print(f"{v:3d}  {k}")
