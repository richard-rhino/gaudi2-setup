# Gaudi2 server setup (Debian 12)

Order:
1. `sudo bash ~/setup/01-base.sh`     base packages, zsh login shell, deps for Habana packages. Reboot if it tells you to.
2. `sudo bash ~/setup/02-gaudi.sh`    Gaudi 1.24.1 driver (DKMS) + hl-smi + runtime libs, from Intel's Ubuntu 22.04 repo.
3. `sudo bash ~/setup/03-docker.sh`   Docker CE + habana container runtime. Log out/in afterwards.
4. `bash ~/setup/04-zsh.sh`           oh-my-zsh, autosuggestions, syntax highlighting, completions, powerlevel10k.
5. `bash ~/setup/05-llama-cpp.sh`     CPU-only llama.cpp (Gaudi has no llama.cpp backend).
6. `bash ~/setup/06-vllm-gaudi.sh`    vLLM + vllm-gaudi plugin in Intel's PyTorch container: the real Gaudi inference path.

Facts that matter:
- Intel officially supports Ubuntu 22.04/24.04 and RHEL 9 only. Debian 12 works by pointing apt at the `jammy` repo, which contains only `habanalabs-*` packages. The DKMS driver is built locally against kernel 6.1.
- Gaudi software release 1.24.1 (build 482), PyTorch 2.11, vllm-gaudi 0.26.0 pair together. Don't mix releases.
- SPI firmware on the cards may need updating to 1.24.0-fw-62.6.2 (`hl-smi` shows the current version; the driver will complain in dmesg if it is too old).
- Docs: https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html and https://github.com/vllm-project/vllm-gaudi

## Debian-specific driver patch (applied 2026-09-09)
Intel's DKMS source fails to build on Debian because `compat/scripts/generate_flags.sh` mis-resolves
Debian's kernel-headers layout (the arch headers Makefile does `include /usr/src/linux-headers-<ver>-common/Makefile`
with an ABSOLUTE path; the script assumed relative). Result: every header probe failed and it tried to
re-implement kernel APIs the 6.1 kernel already has.

Fix: `~/setup/generate_flags.sh.patched` (diff vs `.orig` is ~10 lines). It is installed at
`/usr/src/habanalabs-1.24.1-482/compat/scripts/generate_flags.sh`.
**If you ever reinstall/upgrade habanalabs-dkms or move to a new Gaudi release, re-apply this patch
before `dkms build`.** Packages are on `apt-mark hold` so a plain `apt upgrade` will not touch them.

After a kernel upgrade, DKMS rebuilds automatically (AUTOINSTALL=yes) using the patched source.
Modules autoload at boot via /etc/modules-load.d/habanalabs.conf.

## Status 2026-09-09 (end of first session)
- Driver, hl-smi, Docker, zsh, llama.cpp (CPU): working.
- vLLM containers: `vllm-gaudi` (main branch, needs transformers<5.17) and `vllm-gaudi-stable` (v0.26.0 pair, recommended).
- BLOCKER: HPU compute returns wrong/nondeterministic results (see memory note). Host IOMMU was in translated mode;
  Intel requires passthrough. `intel_iommu=on iommu=pt` added to /etc/default/grub (backup kept). **Reboot required.**
- After reboot: `bash ~/setup/07-verify.sh` (all rel err ~1e-6 => fixed), then `bash ~/setup/08-bench.sh <model> <tp>`.
- If still broken: `sudo apt install -t bookworm-backports linux-image-amd64 linux-headers-amd64` (6.12) and reboot,
  or reinstall Ubuntu 22.04 and run Intel's habanalabs-installer.sh (the officially supported path).

## Status 2026-09-09 (after reboot): FIXED
`intel_iommu=on iommu=pt` resolved the wrong-results problem completely. Also added: hugepages (Intel formula,
/etc/sysctl.d/90-habanalabs-hugepages.conf) and `--ulimit memlock=-1` on the vLLM container (HCCL init failed
for a second process without it). transformers is pinned to 5.16.1 in the container (5.17 breaks vLLM 0.26).
Both are now baked into 02-gaudi.sh / 06-vllm-gaudi.sh / 06-in-container.sh.
