"""8-rank HCCL all-reduce latency vs message size (decode-sized messages are tiny).
Run: PT_HPU_LAZY_MODE=0 torchrun --nproc-per-node 8 allreduce_latency.py"""
import os, time, torch, torch.distributed as dist
import habana_frameworks.torch  # noqa: F401
import habana_frameworks.torch.distributed.hccl  # noqa: F401
dist.init_process_group("hccl"); r = dist.get_rank(); n = dist.get_world_size()
dev = torch.device("hpu")
for numel in [int(s) for s in os.environ.get("SIZES", "2560,20480,163840,1048576,67108864").split(",")]:
    x = torch.ones(numel, dtype=torch.bfloat16, device=dev)
    for _ in range(20): dist.all_reduce(x)
    torch.hpu.synchronize()
    it = 200 if numel < (1 << 22) else 20
    # back-to-back (pipelined) and synchronous (one at a time, like a decode step waiting on each)
    t = time.perf_counter()
    for _ in range(it): dist.all_reduce(x)
    torch.hpu.synchronize(); pipe = (time.perf_counter() - t) / it
    t = time.perf_counter()
    for _ in range(it): dist.all_reduce(x); torch.hpu.synchronize()
    sync = (time.perf_counter() - t) / it
    if r == 0:
        b = numel * 2
        print(f"{b/1024:10.0f} KiB  pipelined {pipe*1e6:9.1f} us  synced {sync*1e6:9.1f} us  busbw {2*(n-1)/n*b/pipe/1e9:6.2f} GB/s", flush=True)
dist.destroy_process_group()
