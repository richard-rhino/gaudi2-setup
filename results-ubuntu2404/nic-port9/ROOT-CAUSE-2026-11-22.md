# Port 9 flapping: root cause and fix (2026-11-22)

**Cause: the unused external port 8 on each card, kept administratively up by netplan.** It is a host configuration
problem. Nothing found here points at the baseboard, OAM seating or firmware.

Status: confirmed with the cards idle and under a 3-minute 8-card all-reduce load. Not yet re-tested with vLLM at
TP=8, and not yet verified across a reboot.

## Mechanism

1. `/etc/netplan/01-dhcp.yaml` applies DHCP to every interface matching `en*`. That includes the Gaudi external
   (scale-out) netdevs `enp*d8`, `enp*d22` and `enp*d23`, so systemd-networkd sets them administratively UP.
2. They have no cables. An open port that never gets a link is reset by the driver about every 30 s
   (`F/W tuning limit - reconfiguring PHY`, `NIC_PHY_FW_TUNING_TIMEOUT_MS` = 30 s).
3. Port 8 (external) and port 9 (internal, card k <-> card k+4) are the two ports of the same NIC macro and share a
   clock. From `drivers/net/ethernet/intel/hbl_cn/gaudi2/gaudi2_cn_phy.c` (driver 1.24.2-481),
   `gaudi2_cn_phy_port_power_up()`:

   > Port 8 is an external port which will usually be brought UP after all the internal ports are UP. Due to macro
   > clock nest dependency, when PHY reset is called for port 8, port 9 (which is internal) is being toggled and might
   > lost stabilization. A W/A to overcome this issue is to reinit port 9 right after.

   So every port 8 retry takes port 9 through a PHY fini/init, on every card, for as long as port 8 is up.
4. Ports 22 and 23 are both external and share a macro with each other, so their retries hurt no internal link. This
   is why only port 9 ever flapped.

All-reduce and all-gather use every internal link, so one link going away every ~30 s stalls the whole collective.
That matches the earlier trace (stalls of 4-16 s every 10-30 s, then a hang).

## Evidence

Driver debug messages, enabled for 45 s with dynamic debug while port 8 was still up (per card, cards 0-7):

```
2  Card N Port 8 lane 0: F/W tuning limit - reconfiguring PHY
2  Card N Port 9: Performing port 9 PHY reinit following port 8 PHY init
2  Card N Port 22 lane 0: F/W tuning limit - reconfiguring PHY
2  Card N Port 23 lane 0: F/W tuning limit - reconfiguring PHY
```

After `ip link set <enp*d8> down` on all 8 cards (02:20:44 UTC), 100 s of the same messages: only ports 22 and 23
still retry. No port 8 reconfig and no port 9 reinit.

| | Port 8 up | Port 8 down |
|---|---|---|
| Port 9 link-downs | 27,057 in 01:20:00-02:20:44; 4,011 in the last 10 min | 0 since 02:20:44, including the load run |
| 5 KiB 8-card all-reduce, p50 | ~220 us | ~220 us |
| Worst single all-reduce | 4-16 s stalls, hang after ~90 s (`logs/allreduce-latency-trace-1120-1953.log`) | 20 ms once in the first second, otherwise 1-10 ms |
| 180 s trace | did not complete | 179 one-second samples, 3,158-4,519 all-reduces in each |

Load run: `PT_HPU_LAZY_MODE=0 torchrun --nproc-per-node 8 experiments-ubuntu2404/allreduce_latency_trace.py 180`,
log in `logs/allreduce-latency-trace-port8down-1122-0222.log`. The trace finished at 02:25:46, but the processes did
not exit and were killed by the 300 s timeout. That teardown hang was not investigated.

## Fix

`setup-ubuntu2404/05-gaudi-external.network`, installed as `/etc/systemd/network/05-gaudi-external.network`:

```ini
[Match]
Driver=habanalabs_en

[Link]
Unmanaged=yes
```

```bash
sudo install -m 644 setup-ubuntu2404/05-gaudi-external.network /etc/systemd/network/
sudo networkctl reload
# networkd does not take down links it already brought up:
for i in $(ls /sys/class/net | grep -E '^enp.*d(8|22|23)$'); do sudo ip link set $i down; done
```

It sorts before netplan's generated `10-netplan-all-en.network`, so networkd leaves all 24 Gaudi external netdevs
alone. The host NICs do not match. Applied on this machine 2026-11-22; `networkctl list` shows the 24 netdevs as
`unmanaged` and the host NICs still `configured`.

## How to check

```bash
ip -br link | grep -E 'd(8|22|23) '          # expect DOWN without the UP flag
networkctl list | grep -E 'd(8|22|23) '      # expect "unmanaged"
sudo journalctl -k --since -10min | grep -c 'Port 9: link down'   # expect 0 under load
```

To see the mechanism directly (prints to the kernel log, turn it off again with `-p`):

```bash
C=/sys/kernel/debug/dynamic_debug/control
echo 'module habanalabs_cn format "Performing port 9 PHY reinit" +p' | sudo tee $C
echo 'module habanalabs_cn format "reconfiguring PHY" +p' | sudo tee $C
```

## Still open

- vLLM at TP=8 has not been re-run, so the token-rate gain is unmeasured.
- Not verified across a reboot.
- The high uncorrectable-FEC counts on port 9 are probably a consequence of the repeated re-inits. The counts on
  other internal ports (for example card 0 port 7, card 1 ports 2 and 3) are not explained by this and were not
  re-measured.
- If the external ports are ever cabled, remove the networkd rule; a port 8 that links up stops retrying.
