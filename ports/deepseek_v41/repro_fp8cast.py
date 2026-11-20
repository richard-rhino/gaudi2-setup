import torch, habana_frameworks.torch.core  # noqa
torch.manual_seed(0)
x = torch.cat([torch.randn(100000) * s for s in (0.01, 1, 30, 200, 400)]).float()
ref = x.to(torch.float8_e4m3fn).float()
for name, f in {"cast e4m3fn": lambda t: t.to(torch.float8_e4m3fn).float(),
                "cast e4m3fn via bf16": lambda t: t.bfloat16().to(torch.float8_e4m3fn).float()}.items():
    try:
        got = f(x.to("hpu")).cpu()
        bad = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref))
        print(f"{name:22s} mismatches={bad.sum().item()}  example={x[bad][:3].tolist()} -> hpu {got[bad][:3].tolist()} cpu {ref[bad][:3].tolist()}")
    except Exception as e:
        print(name, "FAIL", str(e).splitlines()[0][:80])
try:
    e8 = torch.tensor([2.0**k for k in range(-20, 20)]).to(torch.float8_e8m0fnu)
    print("e8m0 cpu ok; hpu:", (e8.to("hpu").float().cpu() == e8.float()).all().item())
except Exception as e:
    print("e8m0 FAIL", str(e).splitlines()[0][:80])
