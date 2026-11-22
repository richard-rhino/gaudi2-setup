# Gaudi2 internal NIC links: port 9 flapping and FEC errors

Status: **root cause found, fix installed; long soak, TP=8 and reboot checks still open** (2026-11-22, see the next section). The rest of the page
is the investigation up to 2026-11-21. Firmware, BIOS and BMC are all at their latest versions. This page collects all
the evidence in one place: what we see, how it hurts performance, what we tried, and how to collect the data again.

Raw data: [`results-ubuntu2404/nic-port9/`](../results-ubuntu2404/nic-port9/), collected by
[`experiments-ubuntu2404/nic_link_report.sh`](../experiments-ubuntu2404/nic_link_report.sh). The script is read-only
and safe to run while vLLM is serving.

## Update 2026-11-22: resets of the unused external port 8 knock out port 9

Found and tested by a second Claude session on this machine; link counts and driver source re-checked afterwards.

- Port 8 is external and has no cable. Netplan's `en*` DHCP rule (`/etc/netplan/01-dhcp.yaml`) brings its netdev
  (`enp*d8`) up. The second session saw the driver then reset that port's PHY about every 30 s.
- Ports 8 and 9 depend on a shared macro clock. `gaudi2_cn_phy.c` (~line 2366) says a PHY reset of port 8 toggles port 9,
  which "might lost stabilization", and calls `port_9_reinit()` after every port 8 reset.
- With all eight d8 netdevs set down at 02:20:44 UTC:

  | | Port 8 up | Port 8 down |
  |---|---|---|
  | Port 9 link-downs | 27,057 (01:20:00-02:20:44) | **0** in the first ~9 min, ~3 of them under load |
  | 5 KiB 8-card all-reduce | p50 ~220 us, 4-16 s stalls, hung after ~90 s | p50 ~235 us, **worst 20 ms over 179 s** |

  Raw: `link-downs-per-minute-1122.txt`, `allreduce-trace-port8down-1122.txt` in `results-ubuntu2404/nic-port9/`.

**Persistent fix, installed 2026-11-22 02:31 UTC:** [05-gaudi-external.network](05-gaudi-external.network) in
`/etc/systemd/network/` keeps networkd off all 24 external netdevs (d8, d22, d23). Mechanism, driver messages and install
steps: [ROOT-CAUSE-2026-11-22.md](../results-ubuntu2404/nic-port9/ROOT-CAUSE-2026-11-22.md).

By hand, until the next reboot (also needed once after installing the rule, which does not take links down):
```bash
for i in /sys/class/net/enp*d8; do sudo ip link set "$(basename "$i")" down; done
```

Still open:
- A longer soak, and Qwen at TP=8.
- The trace processes did not exit afterwards and had to be killed.
- The persistent rule has not been checked across a reboot.
- The uncorrectable FEC counts on other internal ports are not explained by this.

This supersedes the guess in symptom 5 below that port 9 has a physical signal fault.

## System

| | |
|---|---|
| System | Supermicro SYS-820GH-TNR2 (8x Gaudi2 HLS), board X12DPG-OA6-GD2 |
| BIOS | 2.3 (2025-04-14), the latest on Supermicro's site |
| BMC | 01.07.05, the latest (bundle `X12DPG-OA6-GD2_2.3_X1.07.05_SAA1.4.0.zip`) |
| Cards | 8x HL-225 (Gaudi2 OAM), CPLD 0x10 (2023-10-30) |
| Card firmware | `1.24.0-fw-62.6.2-sec-11`, re-flashed 2026-11-20 ([FIRMWARE-UPDATE.md](FIRMWARE-UPDATE.md)). `habanalabs-firmware 1.24.2-481` is the newest in the Habana apt repo and ships this same image. |
| Driver | habanalabs / habanalabs_cn / habanalabs_ib 1.24.2-52a555d (DKMS), kernel 6.8.0-151-generic, Ubuntu 24.04 |

Each Gaudi2 has 24 integrated 100 GbE RoCE v2 ports:
- Ports 8, 22 and 23 are **external** (scale-out, QSFP-DD). They have no cables here, so DOWN is expected.
- The other 21 ports are **internal**: an all-to-all over the baseboard, 3 ports to each of the other 7 cards. HCCL
  uses them for every collective (all-reduce, all-gather, send/recv).

Card index, bus and module (driver "Card N" = Module ID):

| Card | Bus | Serial | Port 9 link-downs this boot (~28 h) |
|---|---|---|---|
| 0 | 43:00.0 | AO05032509 | **216,187** |
| 1 | 44:00.0 | AO05032524 | 16,056 |
| 2 | 19:00.0 | AO05032528 | 5,046 |
| 3 | 1a:00.0 | AO05032486 | **56,934** |
| 4 | cc:00.0 | AO05032479 | 1,317 |
| 5 | cd:00.0 | AO05032480 | 1,234 |
| 6 | b3:00.0 | AO02023865 | 1,210 |
| 7 | b4:00.0 | AO05032504 | 1,231 |

## Symptoms

1. **Only port 9 ever goes down.** All ~299,000 link-down events this boot are on port 9 (`dmesg`:
   `habanalabs 0000:43:00.0: Card 0 Port 9: link down` / `link up`, often several times in one second). No other
   internal port has a single link-down. The previous boot had 1,561; the boot before that, which lasted only 11 min,
   had none.
2. **It happens on every card, and mostly on card 0.** Card 0 accounts for 72% of the events, card 3 for 19%. Snapshots
   of `hl-smi -n link` taken minutes apart show different cards with port 9 DOWN. Earlier it was 19, 1a, 43, 44 and
   b3; later b3, b4, cc and cd; then 43, 44 and 1a.
3. **It only happens while the cards are in use.** Link-downs per hour: ~33,000-40,000 while serving or benchmarking
   (16-19h, 01-05h). Near 0 while idle (06-19h on 11-21, when no vLLM was running). The links are trained and
   monitored only while a process has the NICs open, which fits.
4. **Port 9 seems to connect card k to card k+4.** Within the same second, link-downs pair up as (0,4) 47x, (1,5) 46x,
   (2,6) 43x and (3,7) 46x. That is almost every link-down second of cards 4-7 (49, 46, 43, 46). The cards 0-3 side
   also flaps on its own much more often. These four links cross between the two halves of the baseboard (cards 0-3
   are on bus 19/1a/43/44 behind CPU socket 0, cards 4-7 on b3/b4/cc/cd behind socket 1).
5. **Uncorrectable FEC errors on port 9 and on several other internal ports.** Excerpt from `hl-smi -n stats`
   (cumulative counters; full table in `fec-summary.tsv`):

   | Card | Port | Uncorrectable codewords | Corrected codewords | Corrected symbol errors per lane 0/1/2/3 |
   |---|---|---|---|---|
   | 0 (43) | 9 | **211,948** | 41.8 M | 761k / 3.89M / 172k / 1.50M |
   | 0 (43) | 7 | **273,261** | 4.08 G | 42 / 331 / 154 / 3.6k |
   | 1 (44) | 2 | **207,581** | 3.64 G | |
   | 1 (44) | 3 | **192,323** | 0.88 G | |
   | 6 (b3) | 6 | **160,837** | 6.06 G | |
   | 1 (44) | 9 | 99,115 | 75.5 M | 9.4k / 159k / 158k / 1.05M |
   | 3 (1a) | 9 | 9,782 | 8.07 M | 2.4k / 110k / 12k / 270k |
   | 2 (19) | 9 | 84 | 0.70 M | 0 / 0 / 0 / 0 |
   | typical healthy internal port | | 0 | 0.01-57 G (median 0.66 G) | tens to hundreds |

   33 internal ports have more than 1,000 uncorrectable codewords. So the margin is poor on more links than port 9;
   port 9 on cards 0, 1 and 3 is just bad enough to drop the link. Lanes 1 and 3 of card 0 port 9 have millions of
   corrected symbol errors, while lanes 0 and 2 are much lower, which points at a per-lane signal problem (baseboard
   trace, connector or OAM seating) rather than firmware.

## Impact

- **TP=8 inference is ~10x slower than its compute.** A Qwen3.8-Flash-Next decode step needs ~98 all-reduces. A
  device profile shows ~10 ms of compute per step out of ~300 ms (FINDINGS section 18).
- A decode-sized (5 KiB) 8-card all-reduce has a p50 of ~220 us, but **stalls for 4-16 s every 10-30 s**. After ~90 s
  it hung for good (`experiments-ubuntu2404/allreduce_latency_trace.py`, `logs/allreduce-latency-trace-1120-1953.log`).
- Large all-reduce bandwidth varies a lot between runs: 2-3 GB/s before the re-flash, then 39-42 GB/s, then 199 GB/s
  (128 MB, `allreduce_latency.py`, `logs/allreduce-latency-*.log`).
- **Driver crash** (FINDINGS section 16): a lazy-mode HCCL context created while port 9 was flapping gave a NULL
  dereference in `hbl_ib_eqe_work_schd` (habanalabs_ib) and an endless soft lockup in
  `gaudi2_cn_phy_link_status_work -> do_fw_tuning` (habanalabs_cn). Only a reboot recovered it.
- **Workaround in use: pipeline parallelism.** A point-to-point chain 0->1->...->7 of 20 KiB sends takes ~0.8 ms with
  no stall over 120 s (`p2p_latency_trace.py`, `logs/p2p-latency-trace-1120-2317.log`). PP=8 runs Qwen3.8-Flash-Next
  at ~27 tok/s for a single user, vs 2.6-3.4 tok/s at TP=8. Point-to-point traffic between two cards uses only
  the 3 ports to that peer, so a bad link stalls just one hop. All-reduce and all-gather use every link at once, so one
  bad link stalls all of them. (Which physical card each PP rank lands on is not verified.)

## Effect on pipeline parallelism, and the workaround (2026-11-21)

Point-to-point traffic is affected too, but only on the port-9 pairs. vLLM workers open whichever card is free
first, so the PP rank -> card mapping changes on every start. One start, timed per stage with device events
(`VLLM_PP_STAGE_TIMING=1`):

| PP rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Card (module) | 6 | 4 | **7** | **3** | 2 | 0 | **5** | **1** |

In 3 of 17 windows of 100 decode steps, hop 2->3 (card 7->3) added ~28 ms per token and hop 6->7 (card 5->1)
added ~44 ms. All other hops added nothing. Both are k <-> k+4 (port 9) pairs. Those windows are the slow runs
(16 tok/s instead of 29).

Workaround: `VLLM_HPU_MODULE_ORDER=0,1,2,3,4,5,6,7` (vllm-gaudi patch; `qwen38fn-serve.sh` sets it). Each worker
is spawned with `HLS_MODULE_ID=<i>`, so rank i runs on module i (the `module_id` in
`/sys/class/accel/accelN/device/module_id`, the same as the driver's "Card N"). The PP chain then never uses a port-9
pair and crosses between the two halves of the baseboard once (3->4).

Two things that do not work:
- `HABANA_VISIBLE_MODULES`. The bridge only turns it into `HLS_MODULE_ID` inside `initialize_distributed_hpu()`, which
  vLLM does not call. Without `HLS_MODULE_ID` the device is opened "by type", i.e. the first free card.
- Setting the variable inside the worker's `init_device`. The worker has already opened the device by then.

## What was tried

| Attempt | Result |
|---|---|
| Warm reboot (after the driver crash) | Flapping comes back within ~15 min of load. |
| Re-flash all 8 cards with `hl-fw-loader` (same version) | Large all-reduce bandwidth improved (2-3 -> 39-199 GB/s). Port 9 still flaps. |
| Mask port 9 (`nic_ports_mask=0xFFFDFF` module parameter) | HCCL hangs: it expects the full internal all-to-all. Reverted. |
| Check for newer firmware | None: card fw, BIOS and BMC are all latest (2026-11-21). |
| BMC event log (`ipmitool sel elist`) | 132 entries: only PSU presence at boot and two "Unknown #0xff" on 11-19. Nothing about PCIe or links. |

Not tried yet (list from 2026-11-21, before the cause was found; not needed for the port 9 flapping):
- A **cold power cycle** (AC off for ~1 min, or `ipmitool chassis power cycle`). It resets the baseboard retimers and
  the card CPLDs, which a warm reboot does not.
- Intel's qualification tests for the links: `/opt/habanalabs/qual/gaudi2/bin/hl_qual` (NIC loopback / all-to-all,
  needs exclusive use of the cards).
- Re-seating the OAM modules for cards 0 (43:00.0) and 3 (1a:00.0) and their port-9 partners 4 (cc:00.0) and 7
  (b4:00.0). This needs physical access.
- Older kernel 6.8.0-53 (the DKMS driver is also built for it). Unlikely to help with physical-layer errors.

## How to collect the data again

```bash
# everything (read-only, ~1 min):
experiments-ubuntu2404/nic_link_report.sh results-ubuntu2404/nic-port9-$(date +%m%d)

# quick checks:
for b in 19 1a 43 44 b3 b4 cc cd; do echo -n "$b: "; hl-smi -i 0000:$b:00.0 -n link | grep DOWN | tr '\n' ' '; echo; done
sudo journalctl -k -b 0 | grep "link down" | grep -oE "Card [0-9]+ Port [0-9]+" | sort | uniq -c
hl-smi -i 0000:43:00.0 -n stats -P 9 | grep -E "uncorrect|symbol_err"

# effect on collectives (needs the cards to itself):
~/vllm-venv/bin/torchrun --nproc-per-node 8 experiments-ubuntu2404/allreduce_latency_trace.py
~/vllm-venv/bin/torchrun --nproc-per-node 8 experiments-ubuntu2404/p2p_latency_trace.py
```

The counters in `hl-smi -n stats` are cumulative since the NIC was last initialized. Take two snapshots under load to
get rates.
