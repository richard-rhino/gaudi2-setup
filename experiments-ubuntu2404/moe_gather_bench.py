"""One Qwen3.8-Flash-Next MoE layer (512 experts, H 2560, I 640, top-10), bf16, on one Gaudi2, both compiled.
fused: torch.ops.hpu.mixture_of_experts with per-expert views (as vllm-gaudi passes them)
gather: index_select the routed experts' weights, bmm."""
import time, torch
import habana_frameworks.torch as ht  # noqa: F401
E, H, I, K = 512, 2560, 640, 10
dev = "hpu"
torch.manual_seed(0)
W13 = (torch.randn(E, 2 * I, H, device=dev) * 0.02).to(torch.bfloat16)  # gate | up
W2 = (torch.randn(E, H, I, device=dev) * 0.02).to(torch.bfloat16)
v13 = tuple(W13[e] for e in range(E)); v2 = tuple(W2[e] for e in range(E))

def fused(x, ids, w):
    return torch.ops.hpu.mixture_of_experts(hidden_states=x, expert_routing_table=ids, router_weights=w,
        w12=v13, w3=v2, permuted_weights=True, activation="silu", experts_min=0, experts_max=E - 1)

def gather(x, ids, w):
    T = x.shape[0]
    flat = ids.reshape(-1)
    h = torch.bmm(W13.index_select(0, flat), x.repeat_interleave(K, 0).unsqueeze(-1)).squeeze(-1)  # [TK, 2I]
    a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
    o = torch.bmm(W2.index_select(0, flat), a.unsqueeze(-1)).squeeze(-1)  # [TK, H]
    return (o.view(T, K, H).float() * w.view(T, K, 1).float()).sum(1).to(x.dtype)

def ref(x, ids, w):
    out = torch.zeros(x.shape, dtype=torch.float32)
    xc, Wa, Wb = x.float().cpu(), None, None
    for t in range(x.shape[0]):
        for j in range(K):
            e = int(ids[t, j]); a = W13[e].float().cpu() @ xc[t]
            a = torch.nn.functional.silu(a[:I]) * a[I:]
            out[t] += float(w[t, j]) * (W2[e].float().cpu() @ a)
    return out

cf, cg = torch.compile(fused, backend="hpu_backend"), torch.compile(gather, backend="hpu_backend", dynamic=False)
def bench(f, x, ids, w, it=100):
    for _ in range(3): y = f(x, ids, w)
    torch.hpu.synchronize(); t = time.perf_counter()
    for _ in range(it): y = f(x, ids, w)
    torch.hpu.synchronize(); return (time.perf_counter() - t) / it * 1e3, y

for T in (1, 4, 16, 32, 64):
    x = torch.randn(T, H, device=dev).to(torch.bfloat16)
    w, ids = torch.topk(torch.softmax(torch.randn(T, E, device=dev), -1), K)
    w = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16)
    tg, yg = bench(cg, x, ids, w)
    try:
        tf, yf = bench(cf, x, ids, w) if T <= 4 else (float("nan"), yg)
    except RuntimeError:
        tf, yf = float("nan"), yg
    r = ref(x, ids, w) if T <= 4 else None
    e = lambda y: ((y.float().cpu() - r).abs().max() / r.abs().max()).item() if r is not None else float("nan")
    print(f"T={T:3d}: fused {tf:6.2f} ms (err {e(yf):.1e})   gather {tg:6.2f} ms (err {e(yg):.1e})", flush=True)
