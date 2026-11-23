"""One PP rank: device activity segments (split at idle gaps > GAP ms) in the last decode steps, with busy time,
kernel count and the top op paths; plus hccl recv/send and compileGraph marks. Times relative to the step's recv."""
import collections, json, os, sys
GAP = float(os.getenv("GAP", "0.25")) * 1e3
ev = json.load(open(sys.argv[1]))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
k = sorted((e for e in X if e.get("cat") == "kernel"), key=lambda e: e["ts"])
hc = sorted((e for e in X if e.get("cat") == "hpu_op" and e["name"] in ("hcclRecv", "hcclSend")), key=lambda e: e["ts"])
comp = [e["ts"] for e in X if e["name"] == "compileGraph"]
print("compileGraph events in trace:", len(comp), "| hccl:", [(e["name"][4:], round(e["dur"] / 1e3, 2)) for e in hc][-8:])
segs, cur = [], [k[0]]
for a, b in zip(k, k[1:]):
    if b["ts"] - max(x["ts"] + x["dur"] for x in cur[-40:]) > GAP: segs.append(cur); cur = []
    cur.append(b)
segs.append(cur)
t0 = segs[-int(sys.argv[2]) if len(sys.argv) > 2 else -12][0]["ts"]
prev_end = None
for c in segs[-(int(sys.argv[2]) if len(sys.argv) > 2 else 12):]:
    a, b = c[0]["ts"], max(x["ts"] + x["dur"] for x in c)
    iv = sorted((x["ts"], x["ts"] + x["dur"]) for x in c); busy = last = 0
    for s_, e_ in iv:
        if e_ > last: busy += e_ - max(s_, last); last = e_
    paths = collections.Counter()
    for x in c:
        nm = x.get("args", {}).get("EventName", x["name"]).split("/")
        paths["/".join(nm[:2])] += x["dur"]
    gap = (a - prev_end) / 1e3 if prev_end else 0
    print(f"+{(a - t0) / 1e3:8.2f} ms  idle-before {gap:6.2f}  span {(b - a) / 1e3:6.2f}  busy {busy / 1e3:5.2f}  n {len(c):5d}  "
          + ", ".join(f"{p}:{v / 1e3:.2f}" for p, v in paths.most_common(4)))
    prev_end = b
