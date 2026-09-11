import torch, habana_frameworks.torch as ht, torch.nn.functional as F, os
torch.manual_seed(0)
H="hpu"
def err(name, ref, got): print(f"{name:34s} max|diff| = {(ref.float()-got.cpu().float()).abs().max().item():.3e}")
print("PT_HPU_LAZY_MODE =", os.environ.get("PT_HPU_LAZY_MODE","0"))
q=torch.randn(1,14,8,64); k=torch.randn(1,2,8,64); v=torch.randn(1,2,8,64)
# 1. SDPA variants
err("sdpa causal (GQA expanded)", F.scaled_dot_product_attention(q,k.repeat_interleave(7,1),v.repeat_interleave(7,1),is_causal=True),
    F.scaled_dot_product_attention(q.to(H),k.repeat_interleave(7,1).to(H),v.repeat_interleave(7,1).to(H),is_causal=True))
try:
    err("sdpa causal enable_gqa", F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True),
        F.scaled_dot_product_attention(q.to(H),k.to(H),v.to(H),is_causal=True,enable_gqa=True))
except Exception as e: print("sdpa enable_gqa:", type(e).__name__, str(e)[:80])
m=torch.full((8,8), torch.finfo(torch.float32).min).triu(1)
kk=k.repeat_interleave(7,1); vv=v.repeat_interleave(7,1)
err("sdpa float mask (finfo.min)", F.scaled_dot_product_attention(q,kk,vv,attn_mask=m), F.scaled_dot_product_attention(q.to(H),kk.to(H),vv.to(H),attn_mask=m.to(H)))
bm=torch.ones(8,8,dtype=torch.bool).tril()
err("sdpa bool mask", F.scaled_dot_product_attention(q,kk,vv,attn_mask=bm), F.scaled_dot_product_attention(q.to(H),kk.to(H),vv.to(H),attn_mask=bm.to(H)))
bm4=bm[None,None].expand(1,1,8,8)
err("sdpa bool mask 4D", F.scaled_dot_product_attention(q,kk,vv,attn_mask=bm4), F.scaled_dot_product_attention(q.to(H),kk.to(H),vv.to(H),attn_mask=bm4.to(H)))
# manual attention
def manual(q,k,v,mask):
    s=(q@k.transpose(-1,-2))/8.0; s=s+mask; return torch.softmax(s,-1)@v
err("manual attention", manual(q,kk,vv,m), manual(q.to(H),kk.to(H),vv.to(H),m.to(H)))
# 2. RoPE pieces
x=torch.randn(1,14,8,64)
def rot(x): x1,x2=x[...,:32],x[...,32:]; return torch.cat((-x2,x1),-1)
err("rotate_half (cat/neg/slice)", rot(x), rot(x.to(H)))
inv=1.0/(10000**(torch.arange(0,64,2).float()/64)); pos=torch.arange(8).float()
fr=torch.outer(pos,inv); emb=torch.cat((fr,fr),-1); cos,sin=emb.cos(),emb.sin()
def rope(x,cos,sin): return x*cos[None,None]+rot(x)*sin[None,None]
err("rope apply", rope(x,cos,sin), rope(x.to(H),cos.to(H),sin.to(H)))
invh=1.0/(10000**(torch.arange(0,64,2,device=H).float()/64)); frh=torch.outer(torch.arange(8,device=H).float(),invh)
err("rope freqs computed on hpu", emb, torch.cat((frh,frh),-1)); err("cos on hpu", cos, torch.cat((frh,frh),-1).cos())
# 3. RMSNorm, SiLU-gated MLP, bf16 versions
def rms(x,w,eps=1e-6): return w*(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps))
h=torch.randn(1,8,896); w=torch.rand(896)
err("rmsnorm fp32", rms(h,w), rms(h.to(H),w.to(H)))
err("rmsnorm bf16", rms(h.bfloat16(),w.bfloat16()), rms(h.bfloat16().to(H),w.bfloat16().to(H)))
g=torch.randn(896,4864); u=torch.randn(896,4864); d=torch.randn(4864,896)
err("silu-gated mlp fp32", (F.silu(h@g)*(h@u))@d, (F.silu(h.to(H)@g.to(H))*(h.to(H)@u.to(H)))@d.to(H))
err("silu-gated mlp bf16", ((F.silu(h.bfloat16()@g.bfloat16())*(h.bfloat16()@u.bfloat16()))@d.bfloat16()), ((F.silu(h.bfloat16().to(H)@g.bfloat16().to(H))*(h.bfloat16().to(H)@u.bfloat16().to(H)))@d.bfloat16().to(H)))
# 4. misc
err("triu/full/finfo.min", m, torch.full((8,8), torch.finfo(torch.float32).min, device=H).triu(1))
a=torch.randn(5,7); err("where/arange/gt", torch.where(a>0,a,torch.zeros(1)), torch.where(a.to(H)>0,a.to(H),torch.zeros(1,device=H)))
err("repeat_interleave", k.repeat_interleave(7,1), k.to(H).repeat_interleave(7,1))
err("transpose+reshape", q.transpose(1,2).reshape(1,8,896), q.to(H).transpose(1,2).reshape(1,8,896))
err("bf16 matmul (large K)", (h.bfloat16()@g.bfloat16()).float(), (h.bfloat16().to(H)@g.bfloat16().to(H)).float())
err("bf16 -> fp32 cast chain", h.bfloat16().float(), h.to(H).bfloat16().float())
E=torch.randn(151936,896); ids=torch.tensor([[785,6722,315,9625,374]])
err("big embedding (151936x896)", E[ids], E.to(H)[ids.to(H)])
err("lm_head big matmul", h[:, :5]@E.T, (h[:, :5].to(H)@E.to(H).T))
