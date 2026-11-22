"""Run tests/models/test_initialization.py::test_can_initialize for the given archs. registry.py imports TokenizerMode from
vllm.config, which this vLLM no longer exports; supply it, nothing else changes."""
import importlib, os, sys, traceback
import pytest
import vllm.config
from vllm.config.model import TokenizerMode
vllm.config.TokenizerMode = TokenizerMode
sys.path.insert(0, os.path.expanduser("~/src/vllm-gaudi"))
t = importlib.import_module("tests.models.test_initialization")
# The test's stand-in for _initialize_kv_caches skips resolve_kv_cache_layout, which this vLLM needs first: add it.
from types import SimpleNamespace
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
_patch = t.patch
def _object(target, name, new):
    def with_layout(self, vllm_config):
        specs = self.model_executor.get_kv_cache_specs()
        resolve_kv_cache_layout(vllm_config, self.model_executor.get_supported_kv_cache_layouts(),
                                [s for x in specs for s in x.values()])
        out = new(self, vllm_config)
        if isinstance(out, tuple):  # old interface: (gpu blocks, cpu blocks, config); now the config alone
            vllm_config.cache_config.num_gpu_blocks = out[0]
            out = out[-1]
        return out
    return _patch.object(target, name, with_layout)
t.patch = SimpleNamespace(object=_object)
bad = 0
for arch in sys.argv[1:]:
    mp = pytest.MonkeyPatch()
    try:
        t.test_can_initialize(arch, mp)
        print("CARD pass", arch, flush=True)
    except BaseException as e:
        bad += 1
        print("CARD FAIL", arch, repr(e)[:400], flush=True)
        traceback.print_exc(limit=-6)
    finally:
        mp.undo()
sys.exit(bool(bad))
