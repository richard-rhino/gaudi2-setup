# Qwen/Qwen3.8-Flash-Next (qwen38fn-pp8-pinned-warm) 2026-11-21T23:31:33+00:00

| Scenario | Users | Prompt/answer tok | Per-user tok/s (p50) | Per-user tok/s (p99 user) | TTFT p50 / p99 (s) | Aggregate out tok/s | Total tok/s |
|---|---|---|---|---|---|---|---|
| single | 1 | 128 / 256 | 30.9 | 30.7 | 0.3 / 0.3 | 29.95 | 46.91 |
| chat-10 | 10 | 1024 / 512 | 11.9 | 8.9 | 0.6 / 1.8 | 105.68 | 324.66 |
| chat-16 | 16 | 1024 / 512 | 10.7 | 3.6 | 0.6 / 79.2 | 107.56 | 325.49 |
| agentic-10-24k | 10 | 24576 / 1024 | 11.2 | 9.4 | 3.7 / 19.2 | 100.19 | 2449.45 |
| mixed-16 | 16 | 8192 / 768 | 9.3 | 8.4 | 1.4 / 10.4 | 139.62 | 1648.33 |
