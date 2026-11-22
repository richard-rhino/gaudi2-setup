# Updating (re-flashing) Gaudi2 card firmware

How the firmware on the 8 HL-225 cards was re-flashed on 2026-11-20, with what actually happened. Tested with
Gaudi SW 1.24.2 on Ubuntu 24.04 (kernel 6.8.0-151). Full log of the run: [`logs/fw-flash-1120-1547.log`](../logs/fw-flash-1120-1547.log).

> **Short version:** `sudo modprobe -r habanalabs_ib habanalabs_en habanalabs_cn habanalabs`, then `sudo hl-fw-loader -y`,
> then reload the modules (or reboot). About 45 s per card, run one card at a time, roughly 6 minutes for 8 cards.
> **Do not power off while it runs.**

## When to do it
- `hl-smi -q` shows a card firmware version older than the one in `/lib/firmware/habanalabs/gaudi2/`.
- Intel support asks for it, or you suspect a corrupted flash image (cards that misbehave right after boot).

It is **not** a fix for everything. On this box it was done to chase internal NIC port 9 flapping (see below) and turned
out to rewrite the same version the cards already had.

## 1. Check what the cards run and what the package ships
```bash
# running firmware, per card
hl-smi -q | grep -E "Bus Id|Firmware \[(SPI|FIT)\]"
# firmware shipped by the installed package (this is what hl-fw-loader will flash)
dpkg -l habanalabs-firmware habanalabs-firmware-tools
cd /lib/firmware/habanalabs/gaudi2 && strings gaudi2-boot-fit.itb | grep -oE 'hl-gaudi2-1\.[0-9.]+-fw-[0-9.]+' | sort -u
```
Gotcha: **the firmware build number does not follow the driver version.** Driver package `1.24.2-481` ships firmware
`hl-gaudi2-1.24.0-fw-62.6.2`. Our cards already reported `1.24.0-fw-62.6.2-sec-11` and *were already current*.
A card saying "1.24.0" next to a "1.24.2" driver is not a mismatch. Compare the `fw-62.x.y` part against the image file.

`habanalabs-firmware-tools` (which provides `/usr/sbin/hl-fw-loader`) is installed by `habanalabs-installer.sh`.

## 2. Stop everything that uses the cards and unload the driver
`hl-fw-loader` refuses to run while the driver owns a card:
```
Device: 0000:44:00.0 is in use by the driver, cannot proceed.
Please unbind device: 0000:44:00.0 or unload the habanalabs driver.
```
```bash
pgrep -af "vllm|python|torchrun"     # make sure no workload holds a card
sudo modprobe -r habanalabs_ib habanalabs_en habanalabs_cn habanalabs
lsmod | grep ^habanalabs              # only habanalabs_compat may remain
```

## 3. Flash
```bash
sudo hl-fw-loader -s                  # optional: status / device list
sudo hl-fw-loader -y 2>&1 | tee ~/fw-flash-$(date +%m%d-%H%M).log
```
- With no `-d`, it programs **every** Habana device, one after the other. Use `-d 44:00.0` for a single card.
- With no `-f`, it uses the package's default image under `/lib/firmware/habanalabs/gaudi2/`.
- Each card resets twice and prints `Ok (44.5 seconds)`. The run ends with `#### Finished sending firmware:	Ok`.
- `Missing firmware's version information - skipping version check` is printed for every card and is harmless.
- If a flash is interrupted, `hl-fw-loader -r` (recovery) re-programs the primary image.

Check that every card succeeded:
```bash
grep -E "Sending to|Ok \(" ~/fw-flash-*.log | paste - -    # 8 lines, each ending in Ok
```

## 4. Bring the driver back
All four modules are needed. Loading only `habanalabs` leaves the NIC modules out, and every PyTorch process then fails
with `synStatus=26 [Generic failure] Device acquire failed`:
```bash
sudo modprobe habanalabs
sudo modprobe -a habanalabs_cn habanalabs_en habanalabs_ib
sleep 30 && hl-smi                     # 8 cards, ~768 MiB used each
```
A full reboot does the same thing and is the cleanest option.

## 5. Verify
```bash
hl-smi -q | grep -E "Firmware \[SPI\]" | sort | uniq -c   # all 8 identical
# 8-card HCCL all-reduce check: experiments-ubuntu2404/allreduce.py
PT_HPU_LAZY_MODE=0 torchrun --nproc-per-node 8 experiments-ubuntu2404/allreduce.py
sudo dmesg | grep -oE "Card [0-9] Port [0-9]+: link (up|down)" | sort | uniq -c | sort -rn
```

## What it did on this box
Background: after the driver crash (FINDINGS 16), internal NIC **port 9** flapped (link down/up, hundreds of times)
on the cards as soon as HCCL brought the links up. An 8-card 128 MB bf16 all-reduce was correct but ran at **2–3 GB/s**
bus bandwidth.

| Step | All-reduce busbw (128 MB, 8 cards) | Port 9 link events |
|---|---|---|
| After reboot, before flashing | 2–3 GB/s | 290 on card 1, 8–26 on the others |
| After flashing (same `fw-62.6.2`), driver reloaded | 39–42 GB/s (first run after a reload: 4 GB/s) | 256 on card 1, then 780 on card 0 after another reload; 2–36 on the others |
| Driver with `nic_ports_mask=0xFFFDFF` (port 9 off) | **hangs**, killed by timeout, `unexpected user error interrupt` | none |

Takeaways:
1. All 8 flashes succeeded. The version did not change, because the package carries the same firmware the cards had.
2. The flash plus a full device reset coincided with a big bandwidth improvement, but the port 9 flapping continued.
   The worst card changes between driver reloads (card 1, then card 0), so this does not look like one dead
   transceiver.
3. **Do not mask an internal port on an 8-card HLS-Gaudi2 board.** HCCL expects every internal link (3 per card pair)
   and hangs without it. The `nic_ports_mask` parameter is only usable for the external scale-out ports
   (8, 22, 23). If you tried it, undo it by reloading the driver without the parameter (or by deleting
   `/etc/modprobe.d/habanalabs-ports.conf` if you made it persistent).
4. Measured bandwidth is still well below what this hardware should reach. Next diagnostic step:
   `/opt/habanalabs/qual/gaudi2/bin/hl_qual` NIC tests. The evidence so far is in
   [NIC-LINK-FLAPPING.md](NIC-LINK-FLAPPING.md).
