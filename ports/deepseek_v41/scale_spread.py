"""Measure E8M0 scale spread of DeepSeek-V4.1 FP4 routed experts (numpy only) to quantify FP4->FP8 conversion
exactness on Gaudi2 (E4M3 max 240). Per-row FP8 scale 2^(emax-5) keeps 6*2^5=192<=240; a 32-wide block is exact
iff its scale exponent e >= emax-13 (FP4 values times 2^k stay on the E4M3 grid for k>=-8)."""
import sys, glob, json, numpy as np
import struct
def raw_tensors(path):
    """Minimal safetensors reader: name -> (dtype, shape, uint8 memmap view). Handles F8_E8M0."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = (v["dtype"], v["shape"], mm[8 + n + a: 8 + n + b])
    return out
snap = sys.argv[1]
import os
idx = snap + "/model.safetensors.index.json"
wm = json.load(open(idx if os.path.exists(idx) else sys.argv[2]))["weight_map"]
names = [n for n in wm if ".ffn.experts." in n and n.endswith(".scale") and not n.startswith("mtp.")]
import random; random.seed(0); names = random.sample(names, min(600, len(names)))
by_file = {}
for n in names:
    by_file.setdefault(wm[n], []).append(n)
row_spread, blk128_spread, n_rows = [], [], 0
missing = 0
for f, ns in by_file.items():
    try:
        h = raw_tensors(snap + "/" + f)
    except FileNotFoundError:
        missing += len(ns); continue
    for n in ns:
        dt, shape, buf = h[n]
        s = np.asarray(buf).reshape(shape).astype(np.int32)  # [N, K/32] biased E8M0 exponents
        row_spread.append(s.max(1) - s.min(1))
        N, KB = s.shape
        b = s[: N // 128 * 128].reshape(N // 128, 128, KB // 4, 4).transpose(0, 2, 1, 3).reshape(N // 128, KB // 4, -1)
        blk128_spread.append((b.max(-1) - b.min(-1)).ravel())
rs = np.concatenate(row_spread); bs = np.concatenate(blk128_spread)
print(f"sampled {len(names)-missing} expert scale tensors, {rs.size} rows ({missing} not downloaded yet)")
for name, x in (("per-row (per-channel FP8)", rs), ("128x128 block FP8", bs)):
    print(f"{name:28s} spread(log2) median={np.median(x):.0f} p99={np.percentile(x,99):.0f} max={x.max()}  "
          f"rows/blocks with any lossy sub-block (spread>13): {100*(x>13).mean():.4f}%")
