import torch, habana_frameworks.torch.core  # noqa
cases = {
 "fp32 4D [2,3,64,64]@[64,64]": (torch.float32, (2,3,64,64), (64,64)),
 "fp32 5D [1,2,3,64,64]@[64,64]": (torch.float32, (1,2,3,64,64), (64,64)),
 "fp32 5D@5D": (torch.float32, (1,2,3,64,64), (1,2,3,64,64)),
 "bf16 5D@5D": (torch.bfloat16, (1,2,3,64,64), (1,2,3,64,64)),
 "fp32 4D@4D": (torch.float32, (2,3,64,64), (2,3,64,64)),
 "fp32 3D@3D": (torch.float32, (6,64,64), (6,64,64)),
}
for name, (dt, sa, sb) in cases.items():
    a = torch.randn(sa, dtype=dt); b = torch.randn(sb, dtype=dt)
    try:
        r = (a.to("hpu") @ b.to("hpu")).cpu()
        print(f"OK   {name}  err={((r.float()-(a@b).float()).norm()/(a@b).float().norm()).item():.2e}")
    except Exception as e:
        print(f"FAIL {name}: {str(e).splitlines()[0][:80]}")
