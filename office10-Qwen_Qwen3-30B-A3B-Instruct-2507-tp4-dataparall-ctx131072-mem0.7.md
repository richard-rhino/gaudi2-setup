# Qwen/Qwen3-30B-A3B-Instruct-2507  TP=4 extra=[--data-parallel-size 2]  max_model_len=131072  gpu_mem=0.7 graph_mem=0.2  startup 13 min

| Scenario | Users | Prompt/answer tok | Per-user tok/s (p50) | Per-user tok/s (p99 user) | TTFT p50 / p99 (s) | Aggregate out tok/s | Total tok/s |
|---|---|---|---|---|---|---|---|
| warm | 4 | 1024 / 128 | 15.6 | 13.2 | 0.3 / 0.6 | 51.27 | 499.22 |
| chat-10 | 10 | 1024 / 512 | 17.6 | 17.2 | 0.3 / 1.9 | 159.07 | 488.67 |
| chat-16 | 16 | 1024 / 512 | 17.2 | 16.4 | 0.3 / 2.4 | 248.26 | 751.26 |
| agentic-10-32k | 10 | 32768 / 1024 | 0.0 | 0.0 | 0.0 / 0.0 | FAIL: WARNING 09-10 20:32:52 [interface.py:368] Failed to import from vllm._C: ModuleN |  |
| mixed-16 | 16 | 8192 / 768 |  |  | 0.0 / 0.0 | 0.00 | 0.00 |
| longctx-4-100k | 4 | 100000 / 512 |  |  | 0.0 / 0.0 | 0.00 | 0.00 |
