import torch, kernel_torch as K, kernel_helper as H
torch.manual_seed(0)
ok = lambda c: "PASS" if c else "FAIL"
# 1. E4M3 rounding vs torch's reference cast (incl. exact ties and subnormals)
v = torch.cat([torch.randn(200000) * s for s in (0.001, 0.05, 1, 20, 150, 300)])
ties = torch.tensor([2.0 ** e * (1 + (2 * k + 1) / 16) for e in range(-9, 8) for k in range(8)])
v = torch.cat([v, ties, -ties]).clamp(-448, 448)
ref = v.to(torch.float8_e4m3fn).float()
print("e4m3 rounding     ", ok(torch.equal(K.round_to_e4m3(v), ref)), (K.round_to_e4m3(v) != ref).sum().item())
# 2. E2M1 rounding vs exhaustive RNE on the grid
grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
def rne_e2m1(x):
    a = x.abs().clamp(max=6); d = (a.unsqueeze(-1) - grid).abs(); dmin = d.min(-1, keepdim=True).values
    cand = (d == dmin); even = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)  # even mantissa codes
    pick = torch.where(cand.sum(-1, keepdim=True) > 1, cand & even, cand).float().argmax(-1)
    return torch.copysign(grid[pick], x)
w = torch.cat([torch.randn(100000) * 3, torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0])])
w = torch.cat([w, -w])
print("e2m1 rounding     ", ok(torch.equal(K.round_to_e2m1(w), rne_e2m1(w))))
print("  helper e2m1 ties", [float(H._fp4_decode(H._fp4_encode(torch.tensor([t])))) for t in (.75, 1.75, 3.5)], "(RNE: 1.0, 2.0, 4.0)")
# 3. byte decoders
b = torch.arange(256, dtype=torch.int32).to(torch.uint8)
e4 = b.view(torch.float8_e4m3fn).float(); fin = ~torch.isnan(e4)
print("e4m3 LUT          ", ok(torch.equal(K.decode_e4m3(b).float()[fin], e4[fin])))
print("e8m0 decode       ", ok(torch.equal(K.decode_e8m0(b[1:254]), b[1:254].view(torch.float8_e8m0fnu).float())))
packed = torch.randint(0, 256, (7, 16), dtype=torch.int32).to(torch.uint8)
print("fp4 unpack        ", ok(torch.equal(K.decode_e2m1_packed(packed).float(), H._fp4_storage_to_float(packed))))
# 4. act_quant (pow2 scales) vs helper (torch casts), values with no exact ties
x = torch.randn(64, 256).bfloat16()
mine, s1 = K.act_quant(x, 32, "ue8m0")
hq, hs = H.act_quant(x, 32, "ue8m0")
hdeq = (hq.float().unflatten(-1, (-1, 32)) * hs.float().unsqueeze(-1)).flatten(-2).bfloat16()
print("act_quant fp8     ", ok(torch.equal(mine, hdeq)), (mine != hdeq).sum().item())
# 5. sparse attention vs helper
q = torch.randn(2, 37, 8, 64).bfloat16(); kv = torch.randn(2, 90, 64).bfloat16(); sink = torch.randn(8)
idx = torch.randint(-1, 90, (2, 37, 20)).int(); idx[0, 3] = -1
a1 = K.sparse_attn(q, kv, sink, idx, 0.125, q_chunk=10); a2 = H.sparse_attn(q, kv, sink, idx, 0.125)
print("sparse_attn       ", ok(((a1.float() - a2.float()).abs().max() < 2e-2)), "maxdiff %.4f" % (a1.float() - a2.float()).abs().max(), "empty row zero:", ok(a1[0, 3].abs().max() == 0))
# 6. sinkhorn vs helper
mix = torch.randn(2, 5, 24); scl = torch.rand(3); base = torch.randn(24)
r1 = K.hc_split_sinkhorn(mix, scl, base); r2 = H.hc_split_sinkhorn(mix, scl, base)
print("hc_split_sinkhorn ", ok(all(torch.allclose(a, b, atol=1e-6) for a, b in zip(r1, r2))))
# 7. fp4 weight GEMM path: dequant(fp4) exact vs helper fp4_gemm with identity activations
N_, K_ = 64, 128
wq = torch.randint(0, 256, (N_, K_ // 2), dtype=torch.int32).to(torch.uint8); ws = torch.randint(115, 130, (N_, K_ // 32), dtype=torch.int32).to(torch.uint8)
deq = K.dequant_fp4_weight(wq, ws).float()
ref = (H._fp4_storage_to_float(wq).unflatten(-1, (-1, 32)) * ws.view(torch.float8_e8m0fnu).float().unsqueeze(-1)).flatten(-2)
print("fp4 weight dequant", ok(torch.equal(deq, ref)))
