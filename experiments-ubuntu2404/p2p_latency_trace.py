"""Pipeline-style chain 0->1->...->7 of point-to-point sends (one decode step's hidden state with 4 hyper-connection
streams = 20 KiB), latency per chain per second, to see whether p2p avoids the all-reduce stalls.
Run: PT_HPU_LAZY_MODE=0 torchrun --nproc-per-node 8 p2p_latency_trace.py [seconds]"""
import sys, time, torch, torch.distributed as dist
import habana_frameworks.torch  # noqa: F401
import habana_frameworks.torch.distributed.hccl  # noqa: F401
dist.init_process_group("hccl"); r = dist.get_rank(); n = dist.get_world_size()
x = torch.ones(4 * 2560, dtype=torch.bfloat16, device="hpu")
flag = torch.zeros(1, device="hpu")
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60
for sec in range(int(dur)):
    lat = []; t0 = time.time()
    while True:
        if r == 0:
            stop = torch.tensor([1.0 if time.time() - t0 > 1.0 else 0.0], device="hpu")
        else:
            stop = torch.zeros(1, device="hpu")
        # chain: rank0 decides stop, passes token down the pipe along with data
        t = time.perf_counter()
        if r > 0:
            dist.recv(x, r - 1); dist.recv(stop, r - 1)
        if r < n - 1:
            dist.send(x, r + 1); dist.send(stop, r + 1)
        torch.hpu.synchronize(); lat.append(time.perf_counter() - t)
        if stop.item() > 0:
            break
    if r == n - 1:  # last rank sees full chain latency
        lat.sort()
        print(f"{time.strftime('%H:%M:%S')} n={len(lat):5d} p50={lat[len(lat)//2]*1e6:7.0f}us p90={lat[int(len(lat)*.9)]*1e6:7.0f}us max={lat[-1]*1e3:7.1f}ms", flush=True)
dist.destroy_process_group()
