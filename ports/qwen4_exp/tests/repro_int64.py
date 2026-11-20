import torch, habana_frameworks.torch.core  # noqa
from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers
mult = _build_layer_multipliers(248320, 3, 0, 1234)
print("multipliers", mult.tolist())
torch.manual_seed(0)
tok = torch.randint(0, 248320, (1, 4096), dtype=torch.long)
def h(t, m):
    x = t * m[0]
    x = torch.bitwise_xor(x, torch.roll(t, 1, 1) * m[1])
    return torch.remainder(x, 20000003)
tests = {
  "mul (overflow)": lambda t, m: t * m[0],
  "xor": lambda t, m: torch.bitwise_xor(t * m[0], t * m[1]),
  "remainder": lambda t, m: torch.remainder(t * m[0], 20000003),
  "full hash": h,
  "gather int64": lambda t, m: t.gather(1, torch.arange(4096, device=t.device).flip(0).unsqueeze(0)),
}
for name, f in tests.items():
    ref = f(tok, mult)
    try:
        got = f(tok.to("hpu"), mult.to("hpu")).cpu()
        print(f"{name:16s} exact={torch.equal(ref, got)}  mismatches={(ref != got).sum().item()}")
    except Exception as e:
        print(f"{name:16s} FAIL {str(e).splitlines()[0][:70]}")
