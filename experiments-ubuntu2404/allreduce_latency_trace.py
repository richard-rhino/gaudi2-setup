"""Decode-sized (5 KiB) synced all-reduce latency over time, printed per second, to correlate with NIC link flaps
(compare timestamps with `sudo dmesg -T | grep 'link (up|down)'`).
Run: PT_HPU_LAZY_MODE=0 torchrun --nproc-per-node 8 allreduce_latency_trace.py [seconds]"""
import sys, time, torch, torch.distributed as dist
import habana_frameworks.torch  # noqa: F401
import habana_frameworks.torch.distributed.hccl  # noqa: F401
dist.init_process_group("hccl"); r = dist.get_rank()
x = torch.ones(2560, dtype=torch.bfloat16, device="hpu")
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60
t_end = time.time() + dur
while time.time() < t_end:
    lat = []; t0 = time.time()
    while time.time() - t0 < 1.0:
        t = time.perf_counter(); dist.all_reduce(x); torch.hpu.synchronize(); lat.append(time.perf_counter() - t)
    flag = torch.tensor([1.0], device="hpu"); dist.all_reduce(flag)  # keep ranks in step
    if r == 0:
        lat.sort()
        print(f"{time.strftime('%H:%M:%S')} n={len(lat):5d} p50={lat[len(lat)//2]*1e6:7.0f}us p90={lat[int(len(lat)*.9)]*1e6:7.0f}us max={lat[-1]*1e3:7.1f}ms", flush=True)
dist.destroy_process_group()
