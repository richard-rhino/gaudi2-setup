import sys, torch, habana_frameworks.torch.core  # noqa
sys.path.insert(0, "/home/ubun/work/qwen4"); import hpu_compat; hpu_compat.install_all()
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
fn = M.torch_chunk_gated_delta_rule.__wrapped__ if hasattr(M.torch_chunk_gated_delta_rule, "__wrapped__") else M.torch_chunk_gated_delta_rule
torch.manual_seed(0)
B, T, H, Dk, Dv = 1, 48, 4, 32, 32
q = torch.randn(B, T, H, Dk, dtype=torch.bfloat16); k = torch.randn(B, T, H, Dk, dtype=torch.bfloat16)
v = torch.randn(B, T, H, Dv, dtype=torch.bfloat16)
g = -torch.rand(B, T, H) ; beta = torch.rand(B, T, H, dtype=torch.bfloat16)
ref, _ = fn(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
out, _ = fn(*(x.to("hpu") for x in (q, k, v)), g=g.to("hpu"), beta=beta.to("hpu"), initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
print("rel err", ((out.cpu().float() - ref.float()).norm() / ref.float().norm()).item())
