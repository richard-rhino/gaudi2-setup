# Qwen/Qwen3-30B-A3B-Instruct-2507  TP=8 extra=[]  max_model_len=131072  gpu_mem=0.85 graph_mem=0.1  startup 11 min

| Scenario | Users | Prompt/answer tok | Per-user tok/s (p50) | Per-user tok/s (p99 user) | TTFT p50 / p99 (s) | Aggregate out tok/s | Total tok/s |
|---|---|---|---|---|---|---|---|
| warm | 4 | 1024 / 128 | 26.5 | 23.3 | 0.2 / 0.4 | 87.94 | 856.33 |
| chat-10 | 10 | 1024 / 512 | 28.2 | 27.0 | 0.2 / 0.7 | 251.78 | 773.47 |
| chat-16 | 16 | 1024 / 512 | 29.8 | 27.0 | 0.2 / 1.0 | 423.93 | 1282.89 |
| agentic-10-32k | 10 | 32768 / 1024 | 19.8 | 14.9 | 5.0 / 23.3 | 161.00 | 5194.34 |
| mixed-16 | 16 | 8192 / 768 | 23.5 | 21.9 | 0.6 / 5.3 | 343.00 | 4049.28 |
| longctx-4-100k | 4 | 100000 / 512 | 11.8 | 7.6 | 46.7 / 71.9 | 22.80 | 4790.25 |
