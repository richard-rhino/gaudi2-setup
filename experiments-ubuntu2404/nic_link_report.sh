#!/usr/bin/env bash
# Collect everything about the Gaudi2 scale-up NIC links into one directory.
# Read-only; safe while vLLM is serving. Needs sudo for dmesg/journalctl/dmidecode/ipmitool.
# usage: nic_link_report.sh OUTDIR
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"
BUSES=$(lspci -D | awk '/Habana/{print $1}')
{
  date -u; uname -r; uptime
  sudo dmidecode -s system-manufacturer; sudo dmidecode -s system-product-name; sudo dmidecode -s baseboard-product-name
  echo "BIOS $(sudo dmidecode -s bios-version) $(sudo dmidecode -s bios-release-date)"
  sudo ipmitool mc info 2>/dev/null | grep -A1 -E "Firmware Revision|Aux"
  hl-smi -q | grep -E "Driver Version|Nic Driver|Firmware \[|CPLD Version|Product Name|Serial Number|Bus Id" 
  dpkg -l | awk '/habanalabs/{print $2, $3}'
} > "$OUT/system.txt" 2>&1
for b in $BUSES; do
  s=${b#0000:}; s=${s%%:*}
  { echo "## $b"; hl-smi -i $b -n ports; hl-smi -i $b -n link; } > "$OUT/link-$s.txt" 2>&1
  hl-smi -i $b -n stats > "$OUT/stats-$s.txt" 2>&1
done
# Per-port FEC/error summary table
python3 - "$OUT" <<'PY'
import sys, re, glob, os
out=sys.argv[1]; rows=[]
for f in sorted(glob.glob(f"{out}/stats-*.txt")):
    card=os.path.basename(f)[6:-4]; port=None; d={}
    for line in open(f):
        m=re.match(r"\s*port\s+(\d+):",line)
        if m:
            if port is not None: rows.append((card,port,d))
            port=int(m.group(1)); d={}; continue
        m=re.match(r"\s*(\S+):\s*(-?\d+)",line)
        if m and port is not None: d[m.group(1)]=int(m.group(2))
    if port is not None: rows.append((card,port,d))
keys=["cw_corrected_accum","cw_uncorrect_accum","symbol_err_corrected_lane_0","symbol_err_corrected_lane_1","symbol_err_corrected_lane_2","symbol_err_corrected_lane_3","pre_FEC_SER_int","pre_FEC_SER_exp (negative)","rx_Pkts"]
with open(f"{out}/fec-summary.tsv","w") as w:
    w.write("card\tport\t"+"\t".join(keys)+"\n")
    for c,p,d in rows: w.write(f"{c}\t{p}\t"+"\t".join(str(d.get(k,"")) for k in keys)+"\n")
PY
sudo dmesg -T | grep -E "Port [0-9]+: link (up|down)" > "$OUT/dmesg-link-events.txt"
for bb in -1 0; do
  echo "boot $bb: $(sudo journalctl -k -b $bb 2>/dev/null | grep -cE 'link down')" ; done > "$OUT/link-down-per-boot.txt"
sudo journalctl -k -b 0 | grep -E "link down" | grep -oE "0000:[0-9a-f:.]+ Card [0-9]+ Port [0-9]+" | sort | uniq -c > "$OUT/link-down-per-card-port.txt"
sudo journalctl -k -b 0 | grep -E "link down" | awk '{print $1,$2,$3}' | cut -c1-12 | uniq -c > "$OUT/link-down-per-minute.txt"
sudo dmesg -T | grep -iE "habanalabs" | grep -viE "link (up|down)" | tail -200 > "$OUT/dmesg-habanalabs-other.txt"
echo "wrote $OUT"
