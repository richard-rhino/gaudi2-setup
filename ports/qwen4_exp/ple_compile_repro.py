"""Repro: HPU torch.compile (hpu_backend) miscompiles sigmoid(g) * v.unsqueeze(-2) when g comes from an in-graph
abs/clamp/sqrt/sign chain (Qwen4Exp PLE gate), T=128: "full" and "value_external" are wrong, bmm/cat/einsum are fine.
usage: PT_HPU_LAZY_MODE=0 python ple_compile_repro.py 128  (Gaudi 1.24.2, PyTorch 2.13)"""
import sys, math, torch, torch.nn as nn
import habana_frameworks.torch.core  # noqa
sys.path.insert(0, "/home/ubun/src/vllm-gaudi")
from vllm_gaudi.models.qwen4_exp import GroupedGemmaRMSNorm
dev, dt = "hpu", torch.bfloat16
H, hc = 2560, 4; C = H * hc
T = int(sys.argv[1])
g = lambda s, *sh: torch.randn(*sh, generator=torch.Generator().manual_seed(s)).to(dt).to(dev)
nk, nq = GroupedGemmaRMSNorm(C, H, 1e-6).to(dev), GroupedGemmaRMSNorm(C, H, 1e-6).to(dev)
for n in (nk, nq): n.weight.data.normal_(0, 0.1)
kp = nn.Linear(H, C, bias=False).to(dev, dt); vp = nn.Linear(H, H, bias=False).to(dev, dt)
kp.weight.data.normal_(0, 0.02); vp.weight.data.normal_(0, 0.02)
hidden, emb = g(1, T, C), g(2, T, H)
def gate_of(hidden, emb):
    key = nk(kp(emb)).unflatten(-1, (hc, H)); query = nq(hidden).unflatten(-1, (hc, H))
    gt = (key * query).sum(-1, keepdim=True) / math.sqrt(H)
    return gt.abs().clamp_min(1e-6).sqrt() * gt.sign()
def mul(gt, value, how):
    if how == "bcast": return (torch.sigmoid(gt) * value.unsqueeze(-2)).flatten(-2)
    if how == "bmm":   return torch.bmm(torch.sigmoid(gt), value.unsqueeze(-2)).flatten(-2)  # [T,hc,1]x[T,1,H]
    if how == "cat":   return torch.cat([torch.sigmoid(gt[:, i]) * value for i in range(hc)], dim=-1)
    if how == "einsum": return torch.einsum("tk,th->tkh", torch.sigmoid(gt.squeeze(-1)), value).flatten(-2)
V = {
  "full":            lambda h, e: mul(gate_of(h, e), vp(e), "bcast"),
  "value_external":  None,
  "gate_external":   None,
  "bmm":             lambda h, e: mul(gate_of(h, e), vp(e), "bmm"),
  "cat":             lambda h, e: mul(gate_of(h, e), vp(e), "cat"),
  "einsum":          lambda h, e: mul(gate_of(h, e), vp(e), "einsum"),
  "no_sign":         lambda h, e: (torch.sigmoid((lambda gt: gt)( (nk(kp(e)).unflatten(-1,(hc,H))*nq(h).unflatten(-1,(hc,H))).sum(-1,keepdim=True)/math.sqrt(H))) * vp(e).unsqueeze(-2)).flatten(-2),
}
vv, gg = vp(emb), gate_of(hidden, emb)
V["value_external"] = lambda h, e: mul(gate_of(h, e), vv, "bcast")
V["gate_external"] = lambda h, e: mul(gg, vp(e), "bcast")
for name, f in V.items():
    e = f(hidden, emb); torch._dynamo.reset(); c = torch.compile(f, backend="hpu_backend", dynamic=False)(hidden, emb)
    print(f"T={T} {name:15s} maxdiff {(e.float()-c.float()).abs().max().item():.4g} / {e.float().abs().max().item():.3g}")
