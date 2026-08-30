"""Head-to-head: shipped sm_86 config vs the sm_120 winner vs PyTorch SDPA.

Development tool -- not part of the submission.  Produces the before/after
table for the tech report.
"""
from __future__ import annotations
import gc, importlib.util, os, statistics, sys
import torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT, "torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"] = bm; _spec.loader.exec_module(bm)
import triton

# case -> (B, D, H, S)  and the winner found by tools/tune_attn.py on sm_120
CASES = {1:(64,128,4,128), 8:(64,1024,4,128), 9:(64,128,1,128),
         10:(64,128,2,128), 11:(64,128,16,128), 12:(64,128,4,32), 13:(64,128,4,1024)}
WINNER = {1:(32,32,4,1), 8:(32,16,4,3), 9:(128,32,8,1),
          10:(16,32,4,1), 11:None, 12:None, 13:(128,32,8,2)}

def bench(fn, warmup=15, reps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps): s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a,b in zip(s,e))

def ref_ctx(qkv,B,S,H,hd,scale):
    q,k,v = qkv.view(B,S,3,H,hd).permute(2,0,3,1,4).unbind(0)
    return (F.scaled_dot_product_attention(q.float(),k.float(),v.float(),is_causal=True,scale=scale)
            .transpose(1,2).reshape(B*S,H*hd))

def sdpa_fp16(qkv,B,S,H,hd,scale):
    q,k,v = qkv.view(B,S,3,H,hd).permute(2,0,3,1,4).unbind(0)
    return F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale).transpose(1,2).reshape(B*S,H*hd)

print(f"{'case':>4} {'hd':>4} {'S':>5} {'shipped cfg':>16} {'ship ms':>8} "
      f"{'sm120 cfg':>16} {'best ms':>8} {'sdpa ms':>8} {'gain':>7} {'vs sdpa':>8} {'max_abs':>9}")
for n,(B,D,H,S) in sorted(CASES.items()):
    hd=D//H; scale=hd**-0.5
    torch.manual_seed(1234)
    qkv = torch.randn(B*S,3*D,device="cuda",dtype=torch.float16)*0.5
    ref = ref_ctx(qkv,B,S,H,hd,scale)
    ship = bm._attn_cfgs(S,hd)[0]
    win = WINNER.get(n) or ship
    def run(cfg): return bm._tl_attn(qkv,B,S,H,hd,scale,torch.float16,[cfg])
    try:
        o=run(ship); ship_ms=bench(lambda: run(ship)); ship_err=(o.float()-ref).abs().max().item()
    except Exception as ex: ship_ms=float("nan"); ship_err=float("nan")
    try:
        o=run(win); best_ms=bench(lambda: run(win)); best_err=(o.float()-ref).abs().max().item()
    except Exception: best_ms=float("nan"); best_err=float("nan")
    sd_ms = bench(lambda: sdpa_fp16(qkv,B,S,H,hd,scale))
    print(f"{n:>4} {hd:>4} {S:>5} {str(ship):>16} {ship_ms:>8.4f} {str(win):>16} "
          f"{best_ms:>8.4f} {sd_ms:>8.4f} {ship_ms/best_ms:>6.3f}x {sd_ms/best_ms:>7.3f}x {best_err:>9.2e}")
    del qkv,ref; gc.collect(); torch.cuda.empty_cache()
