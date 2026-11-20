import os, time, torch, torch.distributed as dist
import habana_frameworks.torch.core, habana_frameworks.torch.distributed.hccl  # noqa
dist.init_process_group("hccl"); r = dist.get_rank()
x = torch.full((64 * 1024 * 1024,), float(r + 1), device="hpu", dtype=torch.bfloat16)  # 128 MB
dist.all_reduce(x); torch.hpu.synchronize()
ok = bool((x == 36).all())
t = time.time()
for _ in range(50): dist.all_reduce(x)
torch.hpu.synchronize(); dt = (time.time() - t) / 10
if r == 0: print(f"allreduce ok={ok} 128MB {dt*1e3:.1f} ms busbw={2*7/8*x.numel()*2/dt/1e9:.0f} GB/s", flush=True)
dist.destroy_process_group()
