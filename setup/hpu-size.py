import torch, habana_frameworks.torch as ht, torch.nn.functional as F, os
torch.manual_seed(0); H="hpu"
def rel(ref,got): ref=ref.float(); got=got.cpu().float(); return ((ref-got).abs().max()/ref.abs().max()).item()
print("== roundtrip exactness by size ==")
for mb in (256,512,1024,2048,3072):
    a=torch.randn(mb*1024*1024//4); b=a.to(H).cpu(); print(f"{mb:5d} MB roundtrip exact: {torch.equal(a,b)}  mismatching elems: {(a!=b).sum().item()}")
print("== gather from table by table size ==")
for rows in (1000,20000,50000,100000,151936):
    E=torch.randn(rows,896); ids=torch.randint(0,rows,(64,)); print(f"gather rows={rows:7d} ({rows*896*4/1e6:6.0f} MB) rel err {rel(E[ids],E.to(H)[ids.to(H)]):.2e}")
print("== matmul rel err by shape (fp32) ==")
for (M,K,N) in ((8,896,4864),(64,896,4864),(8,4864,896),(2048,2048,2048),(8,896,151936),(256,896,151936)):
    x=torch.randn(M,K); y=torch.randn(K,N); print(f"({M},{K},{N}) rel err {rel(x@y,x.to(H)@y.to(H)):.2e}")
print("== determinism: same op 5x on device ==")
x=torch.randn(8,896); g=torch.randn(896,4864); d=torch.randn(4864,896)
ref=(F.silu(x@g)*(x@g))@d; outs=[]
for i in range(5):
    o=((F.silu(x.to(H)@g.to(H))*(x.to(H)@g.to(H)))@d.to(H)).cpu(); outs.append(o); print(f"run {i}: rel err vs cpu {rel(ref,o):.2e}  vs run0 {rel(outs[0],o):.2e}")
print("== which stage of the mlp breaks ==")
a_cpu=x@g; a_hpu=(x.to(H)@g.to(H)); print("stage1 x@g", f"{rel(a_cpu,a_hpu):.2e}")
s_cpu=F.silu(a_cpu); s_hpu=F.silu(a_hpu); print("stage2 silu", f"{rel(s_cpu,s_hpu):.2e}")
p_cpu=s_cpu*a_cpu; p_hpu=s_hpu*a_hpu; print("stage3 mul", f"{rel(p_cpu,p_hpu):.2e}")
o_cpu=p_cpu@d; o_hpu=p_hpu@d.to(H); print("stage4 @d (hpu chain)", f"{rel(o_cpu,o_hpu):.2e}")
print("stage4 @d (cpu intermediate moved to hpu)", f"{rel(o_cpu, p_cpu.to(H)@d.to(H)):.2e}")
print("stage4 @d (hpu intermediate moved via cpu)", f"{rel(o_cpu, p_hpu.cpu().to(H)@d.to(H)):.2e}")
