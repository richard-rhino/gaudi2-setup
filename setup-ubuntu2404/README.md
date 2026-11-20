# Ubuntu 24.04 host setup (second Gaudi2 server, 2026-11-20)

Host: 8x Gaudi2 HL-225, 2x Xeon Platinum 8380 (160 threads), 503 GB RAM, 1.7 TB SATA (OS) + 2.9 TB NVMe (`/models`).
Ubuntu 24.04.5 is an officially supported OS, so Intel's `habanalabs-installer.sh` is used directly (no Debian patches needed).
Stack: Gaudi SW 1.24.2, Intel PyTorch 2.13, vLLM 0.34.0rc0 (commit pinned by vllm-gaudi's `VLLM_STABLE_COMMIT`), vllm-gaudi main
(0.27.0rc1.dev199). No Docker: everything runs natively in `~/vllm-venv`.

Order:
1. `sudo bash 01-root-setup.sh`      apt packages, zsh (+autosuggestions/syntax-highlighting from apt), NVMe -> /models via fstab,
                                     render group, `iommu=pt intel_iommu=on`, Gaudi base (driver/firmware/runtime), chsh. Reboot.
2. `sudo bash 02-rebuild-driver.sh`  only if the kernel changed during step 1 (see pitfalls). Also installs the PyTorch deps
                                     (`--type dependencies`, which compiles Open MPI 5.0.8 for ~5-10 min). Reboot.
3. `bash 03-user-stack.sh`           Intel PyTorch venv + vLLM (pinned) + vllm-gaudi.
4. `sudo bash 04-host-tuning.sh`     hugepages, unlimited memlock/nofile, no core dumps.
Shell: copy `zshrc` -> `~/.zshrc`, `zshenv` -> `~/.zshenv` (the zshenv is REQUIRED, see pitfalls). `ask-llm` is a tiny CLI for
the LAN's Nvidia-hosted models (used as helpers during the model ports).

## Pitfalls hit on this install (each one cost time)
- **Kernel bumped under the driver.** `apt-get install linux-headers-$(uname -r)` + later `linux-headers-generic` pulled a newer
  kernel (6.8.0-53 -> -149 -> -151). DKMS had only built habanalabs for the old kernel, so the new kernel booted with the
  *in-tree* upstream habanalabs driver: hl-smi shows the kernel version as "Driver Version" and N/A for every metric.
  Fix: `dkms install habanalabs/1.24.2-481 -k $(uname -r)` + reboot (02-rebuild-driver.sh). With `linux-headers-generic`
  installed, future kernels get headers and DKMS rebuilds automatically.
- **Installer venv path bug.** `HABANALABS_VIRTUAL_DIR` is always prefixed with `$HOME`; an absolute path ends up at
  `/home/u/home/u/venv`. Pass a relative name.
- **zsh doesn't read /etc/profile.d** on Ubuntu. The Gaudi env (`GC_KERNEL_PATH` etc.) is never set, and *every* HPU op fails
  with "Graph compile failed ... synStatus 26". `zshenv` sources `/etc/profile.d/habanalabs*.sh`.
- **`--type pytorch` refuses to run** until `--type dependencies` has completed, including the Open MPI source build. An
  interrupted dependencies run leaves no MPI and the pytorch install fails its pre-check ("One or more dependencies were not detected").
- The old models on the NVMe were root-owned (downloaded from Docker as root); `chown -R` the HF cache.

- Firmware re-flash how-to and port 9 results: [FIRMWARE-UPDATE.md](FIRMWARE-UPDATE.md)
