"""Streaming version of DeepSeek-V4.1's `inference/convert.py`, for hosts with less RAM than the checkpoint.

The upstream converter keeps every rank's full state dict in memory until the end (~510 GB for V4.1-Flash; this host
has 503 GB). This one applies the same renames, sharding and post-processing, but flushes each rank's tensors to
`model{rank}-mp{mp}-part{NNN}.safetensors` once they exceed --flush-gb. Load the parts with
`hpu_patches.load_checkpoint(model, <dir>, rank, mp)`.

  python -I convert_stream.py --inference-dir <snap>/inference --hf-ckpt-path <snap> --save-path /models/dsv41-mp4 \
         --model-parallel 4 --expert-dtype fp4
"""
import argparse
import importlib.util
import json
import os
import shutil
import time
from glob import glob

import torch
from safetensors.torch import safe_open, save_file


def load_upstream(inference_dir: str):
    spec = importlib.util.spec_from_file_location("ds_convert", os.path.join(inference_dir, "convert.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rename(name: str, mapping: dict):
    """The upstream name/key transform; returns (new name, shard dim or None)."""
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    new_key, dim = mapping.get(key, (key, None))
    return name.replace(key, new_key), dim


def shard(name, param, i, mp, dim, n_experts, mtp_n_experts):
    """Rank i's piece of `param`, or None if rank i does not hold it (upstream rules)."""
    if "experts" in name and "shared_experts" not in name:
        n_local = (mtp_n_experts if name.startswith("mtp.") else n_experts) // mp
        idx = int(name.split(".")[-3])
        return param if i * n_local <= idx < (i + 1) * n_local else None
    if ".engram.embed." in name:
        size = (param.size(0) + mp - 1) // mp
        p = param[i * size:(i + 1) * size].contiguous()
        if p.size(0) < size:
            p = torch.cat([p, param.new_full((size - p.size(0), param.size(1)), 1 if name.endswith(".scale") else 0)])
        return p
    if dim is not None:
        assert param.size(dim) % mp == 0, f"{name}: dim {dim} not divisible by {mp}"
        size = param.size(dim) // mp
        return param.narrow(dim, i * size, size).contiguous()
    return param


def postprocess(sd: dict, pending: dict, up, expert_dtype: str):
    """Upstream post-processing on a part; wo_a weight/scale pairs that are not both present yet wait in `pending`."""
    for name in list(sd):
        if name.endswith("wo_a.weight") or name.endswith("wo_a.scale"):
            pending[name] = sd.pop(name)
    for wname in [n for n in pending if n.endswith("wo_a.weight")]:
        sname = wname.replace("weight", "scale")
        if sname not in pending:
            continue
        weight, scale = pending.pop(wname), pending.pop(sname)
        ob, ib = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
        assert (ob, ib) in ((32, 32), (128, 128)), (wname, weight.shape, scale.shape)
        w = weight.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * scale[:, None, :, None].float()
        sd[wname] = w.flatten(2, 3).flatten(0, 1).bfloat16()
    for name in list(sd):
        if "experts" in name and sd[name].dtype == torch.int8 and name.endswith("weight"):
            if expert_dtype == "fp8":
                sname = name.replace("weight", "scale")
                if sname not in sd:
                    pending[name] = sd.pop(name)
                    continue
                sd[name], sd[sname] = up.cast_e2m1fn_to_e4m3fn(sd[name], sd[sname])
            else:
                sd[name] = sd[name].view(torch.float4_e2m1fn_x2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-dir", required=True)
    ap.add_argument("--hf-ckpt-path", required=True)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--model-parallel", type=int, required=True)
    ap.add_argument("--expert-dtype", choices=["fp8", "fp4"], default="fp4")
    ap.add_argument("--flush-gb", type=float, default=8.0)
    a = ap.parse_args()
    up = load_upstream(a.inference_dir)
    torch.set_num_threads(8)
    mp = a.model_parallel
    os.makedirs(a.save_path, exist_ok=True)
    names = set(json.load(open(os.path.join(a.hf_ckpt_path, "model.safetensors.index.json")))["weight_map"])
    n_experts, mtp_n_experts = up.infer_num_experts(names)
    assert n_experts % mp == 0 and mtp_n_experts % mp == 0
    print(f"{n_experts=} {mtp_n_experts=} mp={mp}", flush=True)
    sds = [{} for _ in range(mp)]
    sizes = [0] * mp
    parts = [0] * mp
    pending = [{} for _ in range(mp)]
    seen = set()

    def flush(i, final=False):
        postprocess(sds[i], pending[i], up, a.expert_dtype)
        if final:
            assert not pending[i], f"rank {i}: unpaired tensors {list(pending[i])[:4]}"
        if sds[i]:
            save_file(sds[i], os.path.join(a.save_path, f"model{i}-mp{mp}-part{parts[i]:03d}.safetensors"))
            parts[i] += 1
        sds[i].clear()
        sizes[i] = 0

    files = sorted(glob(os.path.join(a.hf_ckpt_path, "*.safetensors")))
    t0 = time.time()
    for n, fp in enumerate(files):
        with safe_open(fp, framework="pt", device="cpu") as f:
            for src in f.keys():
                seen.add(src)
                name = src[len("model."):] if src.startswith("model.") else src
                if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
                    continue
                param = f.get_tensor(src)
                new_name, dim = rename(src, up.mapping)
                for i in range(mp):
                    p = shard(new_name, param, i, mp, dim, n_experts, mtp_n_experts)
                    if p is None:
                        continue
                    sds[i][new_name] = p
                    sizes[i] += p.numel() * p.element_size()
        for i in range(mp):
            if sizes[i] > a.flush_gb * 2**30:
                flush(i)
        print(f"[{n + 1}/{len(files)}] {os.path.basename(fp)} {time.time() - t0:.0f}s parts={parts}", flush=True)
    for i in range(mp):
        flush(i, final=True)
    assert seen == names, f"{len(names - seen)} missing, {len(seen - names)} unexpected"
    for fn in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(a.hf_ckpt_path, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(a.save_path, fn))
    print(f"done: {parts} parts per rank in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
