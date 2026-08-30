"""Which SDPA backend does this build actually use, and does the custom
attention kernel still earn its place against it?

Development tool -- not part of the submission.

`_attn_kernel`'s docstring justifies its existence with:

    "This PyTorch build has no FlashAttention (Windows wheels ship with
     USE_FLASH_ATTENTION=OFF -- verified: can_use_flash_attention() is False
     for every head_dim), so SDPA always lands on the CUTLASS memory-efficient
     kernel, which is instantiated with kMaxK=64 and therefore wastes ~8x of
     its accumulator width at head_dim=8"

That is a property of Windows wheels, not of the algorithm.  On this Linux
cu128 build `can_use_flash_attention()` returns True for fp16 at every
head_dim in the matrix, including 256.  So the kernel's competitor here is a
real FlashAttention-2 implementation, not a hobbled CUTLASS fallback, and the
comparison has to be redone.

Forces each backend explicitly rather than trusting the dispatcher.
"""
from __future__ import annotations
import gc, importlib.util, os, statistics, sys
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT,"torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"]=bm; _spec.loader.exec_module(bm)

CASES = {1:(64,128,4,128), 7:(64,32,4,128), 8:(64,1024,4,128), 9:(64,128,1,128),
         10:(64,128,2,128), 11:(64,128,16,128), 12:(64,128,4,32), 13:(64,128,4,1024)}

def bench(fn, warmup=15, reps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps): s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a,b in zip(s,e))

def sdpa(qkv,B,S,H,hd,scale):
    q,k,v = qkv.view(B,S,3,H,hd).permute(2,0,3,1,4).unbind(0)
    return F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)\
            .transpose(1,2).reshape(B*S,H*hd)

print(f"{'case':>5} {'hd':>4} {'S':>5} {'triton':>9} {'flash':>9} {'mem-eff':>9} "
      f"{'default':>9} {'triton vs flash':>16} {'max_abs':>10}")
for n,(B,D,H,S) in sorted(CASES.items()):
    hd=D//H; scale=hd**-0.5
    torch.manual_seed(1234)
    qkv=torch.randn(B*S,3*D,device="cuda",dtype=torch.float16)*0.5
    q,k,v = qkv.view(B,S,3,H,hd).permute(2,0,3,1,4).unbind(0)
    ref=(F.scaled_dot_product_attention(q.float(),k.float(),v.float(),is_causal=True,scale=scale)
         .transpose(1,2).reshape(B*S,H*hd))
    cfgs = bm._attn_cfgs(S,hd)
    t_ms = bench(lambda: bm._tl_attn(qkv,B,S,H,hd,scale,torch.float16,cfgs))
    err  = (bm._tl_attn(qkv,B,S,H,hd,scale,torch.float16,cfgs).float()-ref).abs().max().item()
    res={}
    for label,be in (("flash",SDPBackend.FLASH_ATTENTION),
                     ("memeff",SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel(be):
                sdpa(qkv,B,S,H,hd,scale)
                res[label]=bench(lambda: sdpa(qkv,B,S,H,hd,scale))
        except Exception:
            res[label]=float("nan")
    res["default"]=bench(lambda: sdpa(qkv,B,S,H,hd,scale))
    fl=res["flash"]
    print(f"{n:>5} {hd:>4} {S:>5} {t_ms:>9.4f} {res['flash']:>9.4f} {res['memeff']:>9.4f} "
          f"{res['default']:>9.4f} {(fl/t_ms if fl==fl else float('nan')):>15.3f}x {err:>10.2e}")
    del qkv,ref,q,k,v; gc.collect(); torch.cuda.empty_cache()
