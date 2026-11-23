"""Experiment only: host wall time of runner / model phases per step, per process, median of every 200 calls."""
import importlib.abc, os, statistics, sys, time

ACC, GAP = {}, {"last_end": None}

def _log(name, dt):
    v = ACC.setdefault(name, [])
    v.append(dt)
    if len(v) >= 200:
        try:
            import torch.distributed as d
            r = d.get_rank() if d.is_initialized() else -1
        except Exception:
            r = -1
        q = sorted(v)
        print(f"HOSTTIME rank {r} {name}: median {statistics.median(v) * 1e3:.3f} ms p10 {q[len(q) // 10] * 1e3:.3f} "
              f"p90 {q[9 * len(q) // 10] * 1e3:.3f}", file=sys.stderr, flush=True)
        v.clear()

def wrap(cls, meth, name, gap=False):
    f = getattr(cls, meth)
    def w(*a, **k):
        t = time.perf_counter()
        if gap and GAP["last_end"] is not None:
            _log("gap_between_" + name, t - GAP["last_end"])
        try:
            return f(*a, **k)
        finally:
            e = time.perf_counter()
            _log(name, e - t)
            if gap:
                GAP["last_end"] = e
    w.__wrapped__ = f
    setattr(cls, meth, w)

HOOKS = {
    "vllm_gaudi.v1.worker.hpu_model_runner": lambda m: [
        wrap(m.HPUModelRunner, "sample_tokens", "sample_tokens", gap=True),
        wrap(m.HPUModelRunner, "execute_model", "execute_model"),
        wrap(m.HPUModelRunner, "_prepare_inputs", "_prepare_inputs"),
        wrap(m.HPUModelRunner, "_execute_model_generic", "_execute_model_generic"),
        wrap(m.HPUModelRunner, "_run_sampling", "_run_sampling"),
        wrap(m.HpuModelAdapter, "forward", "adapter_forward"),
        wrap(m.HPUModelRunner, "propose_draft_token_ids", "propose_draft_token_ids"),
        wrap(m.HPUModelRunner, "_prepare_spec_decode_inputs", "_prepare_spec_decode_inputs"),
    ],
    "vllm_gaudi.v1.spec_decode.hpu_eagle": lambda m: [wrap(m.HpuEagleProposer, "propose", "drafter_propose")],
    "vllm_gaudi.v1._vendored.sample.rejection_sampler": lambda m: [wrap(m.RejectionSampler, "forward", "rejection_sampler")],
    "vllm.distributed.parallel_state": lambda m: [wrap(m.GroupCoordinator, "send", "pp_send")],
    "vllm_gaudi.ops.hpu_rotary_embedding": lambda m: [
        wrap(c, "prepare_cos_sin", "rope_prepare_cos_sin") for c in vars(m).values()
        if isinstance(c, type) and "prepare_cos_sin" in vars(c)],
    "vllm_gaudi.models.qwen4_exp": lambda m: [
        wrap(m.HpuQwen4ExpModel, "_ple_prepare", "_ple_prepare"),
        wrap(m.HpuQwen4ExpModel, "forward", "model_forward"),
        wrap(m.HpuQwen4ExpDecoderLayer, "forward", "layer_forward"),
    ],
}

class Finder(importlib.abc.MetaPathFinder):
    busy = False
    def find_spec(self, name, path, target=None):
        if self.busy or name not in HOOKS:
            return None
        from importlib.util import find_spec
        self.busy = True
        try:
            spec = find_spec(name)
        finally:
            self.busy = False
        ex = spec.loader.exec_module
        def exec_module(mod):
            ex(mod)
            HOOKS[name](mod)
        spec.loader.exec_module = exec_module
        return spec

sys.meta_path.insert(0, Finder())
