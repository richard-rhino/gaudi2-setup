"""Per rank: device-activity clusters (= this stage's part of each step), split at idle gaps > 8 ms.
Span = first kernel start to last kernel end; busy = union; op types by summed op-instance spans. Plus host compiles."""
import collections, json, sys, statistics as st

def union(iv):
    t = last = 0.0
    for a, b in sorted(iv):
        if b <= last: continue
        t += b - max(a, last); last = b
    return t

allspan = []
for path in sys.argv[1:]:
    ev = json.load(open(path))["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    k = sorted((e for e in X if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    comp = [e for e in X if e["name"] == "compileGraph"]
    cl, cur = [], [k[0]]
    for a, b in zip(k, k[1:]):
        if b["ts"] - max(x["ts"] + x["dur"] for x in cur[-50:]) > 8000:
            cl.append(cur); cur = []
        cur.append(b)
    cl.append(cur)
    cl = cl[-8:]  # the 8 decode steps (the prefill cluster comes first)
    spans, busys, types = [], [], collections.Counter()
    for c in cl:
        a, b = c[0]["ts"], max(x["ts"] + x["dur"] for x in c)
        spans.append((b - a) / 1e3); busys.append(union([(x["ts"], x["ts"] + x["dur"]) for x in c]) / 1e3)
        ops = collections.defaultdict(list)
        for x in c:
            nm = x.get("args", {}).get("EventName", x["name"])
            ops[(x.get("args", {}).get("recipeId"), nm)].append((x["ts"], x["ts"] + x["dur"]))
        for (rid, nm), iv in ops.items():
            types[nm.split("/")[0]] += union(iv)
    s, bz = st.median(spans), st.median(busys)
    allspan.append(s)
    print(f"## {path.split('-')[-1]}: span {s:5.2f} ms, busy {bz:5.2f} ms ({bz / s:.0%}), compiles in trace {len(comp)} "
          f"({sum(e['dur'] for e in comp) / 1e3:.1f} ms host)")
    print("    " + ", ".join(f"{t} {v / len(cl) / 1e3:.2f}" for t, v in types.most_common(8)))
print(f"sum of stage spans {sum(allspan):.1f} ms")
