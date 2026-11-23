"""One card: Qwen expert shapes (512 x [1280, 2560] / [2560, 640]), 1 token, top-10. Device time per layer of
MoE variants, 6 layers' worth of distinct weights, fresh random routing every call. Checks each vs the reference."""
import os, time, torch, habana_frameworks.torch as ht
torch.manual_seed(0)
E_, H, I, K, NL = 512, 2560, 640, 10, int(os.getenv("NL", "6"))
dev = "hpu"
w13 = [torch.randn(E_, 2 * I, H, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(NL)]
w2 = [torch.randn(E_, H, I, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(NL)]
x = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
ids = [torch.stack([torch.randperm(E_)[:K] for _ in range(1)]).to(dev) for _ in range(64)]
wts = torch.softmax(torch.randn(1, K), -1).to(dev)

def ref(l, x, local, wt):  # current: gather weights, bmm
    xk = x.repeat_interleave(K, 0).unsqueeze(-1)
    h = torch.bmm(w13[l].index_select(0, local), xk).squeeze(-1)
    a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
    o = torch.bmm(w2[l].index_select(0, local), a.unsqueeze(-1)).squeeze(-1)
    return (o.float() * wt.view(-1, 1)).sum(0, keepdim=True).to(x.dtype)

def tpc(l, x, local, wt):  # gather fused into an elementwise-multiply + reduce (TPC), fp32 accumulate
    h = (w13[l].index_select(0, local).float() * x.float().view(1, 1, H)).sum(-1)  # [K, 2I]
    a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
    o = (w2[l].index_select(0, local).float() * a.view(K, 1, I)).sum(-1)  # [K, H]
    return (o * wt.view(-1, 1)).sum(0, keepdim=True).to(x.dtype)

def tpc_bf16(l, x, local, wt):
    h = (w13[l].index_select(0, local) * x.view(1, 1, H)).float().sum(-1)
    a = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(x.dtype)
    o = (w2[l].index_select(0, local) * a.view(K, 1, I)).float().sum(-1)
    return (o * wt.view(-1, 1)).sum(0, keepdim=True).to(x.dtype)

def flat_mm(l, x, local, wt):  # one matmul over the gathered rows as a 2D [K*2I, H] @ [H, 1]
    rows = w13[l].index_select(0, local).view(K * 2 * I, H)
    h = (rows @ x.view(H, 1)).view(K, 2 * I)
    a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]  # [K, I]
    o = torch.bmm(w2[l].index_select(0, local), a.unsqueeze(-1)).squeeze(-1)
    return (o.float() * wt.view(-1, 1)).sum(0, keepdim=True).to(x.dtype)

def xT_bmm(l, x, local, wt):  # x as the left operand: [K, 1, H] @ [K, H, 2I]^T -> uses transposed weights view
    xk = x.view(1, 1, H).expand(K, 1, H)
    h = torch.bmm(xk, w13[l].index_select(0, local).transpose(1, 2)).view(K, 2 * I)
    a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
    o = torch.bmm(a.view(K, 1, I), w2[l].index_select(0, local).transpose(1, 2)).view(K, H)
    return (o.float() * wt.view(-1, 1)).sum(0, keepdim=True).to(x.dtype)

VARIANTS = dict(ref=ref, tpc=tpc, tpc_bf16=tpc_bf16, flat_mm=flat_mm, xT_bmm=xT_bmm)
sel = os.getenv("ONLY", "").split(",") if os.getenv("ONLY") else list(VARIANTS)
base = None
for name in sel:
    f = VARIANTS[name]
    cf = torch.compile(lambda l, x, local, wt, f=f: f(l, x, local, wt), backend="hpu_backend", dynamic=False)
    def run_all(it):
        y = x
        for l in range(NL):
            y = cf(l, y, ids[(it * NL + l) % 64].view(-1), wts.view(-1)) + x
        return y
    for it in range(3): run_all(it)
    torch.hpu.synchronize()
    n = 30
    t = time.perf_counter()
    for it in range(n): y = run_all(it)
    torch.hpu.synchronize()
    ms = (time.perf_counter() - t) / n / NL * 1e3
    # correctness vs ref on one layer
    out = cf(0, x, ids[0].view(-1), wts.view(-1)).float(); r = ref(0, x, ids[0].view(-1), wts.view(-1)).float()
    err = ((out - r).norm() / r.norm()).item()
    print(f"{name:10s} {ms:6.3f} ms/layer (wall, incl. launch)  rel err vs ref {err:.2e}", flush=True)
