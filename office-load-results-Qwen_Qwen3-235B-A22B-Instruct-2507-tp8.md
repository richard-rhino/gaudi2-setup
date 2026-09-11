# Office load test: Qwen/Qwen3-235B-A22B-Instruct-2507, TP=8 cards=all extra=[], max_model_len 32768

| Scenario | Requests | Concurrency | Input/Output tokens | Req/s | Output tok/s | Total tok/s | TTFT p50/p99 (s) | TPOT p50/p99 (ms) | E2E p50/p99 (s) |
|---|---|---|---|---|---|---|---|---|---|
| warmup-run | 16 | 8 | 512 / 128 | 0.69 | 88.86 | 449.55 | 0.45 / 1.47 | 80.73 / 94.91 | 10.8 / 14.6 |
| burst-64-at-once | 64 | inf | 1024 / 256 | 1.69 | 432.81 | 2187.05 | 7.54 / 13.10 | 104.55 / 132.88 | 34.1 / 37.9 |
| sustained-32 | 128 | 32 | 2048 / 512 | 0.64 | 323.79 | 1640.10 | 0.54 / 10.12 | 87.29 / 97.88 | 46.0 / 62.1 |
| sustained-64 | 192 | 64 | 2048 / 512 | 1.00 | 510.26 | 2567.35 | 0.93 / 20.32 | 107.64 / 131.10 | 57.0 / 82.9 |
