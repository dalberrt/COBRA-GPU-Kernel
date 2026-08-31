"""In-graph per-kernel breakdown of a small case.

Development tool -- not part of the submission.

The small shapes are latency-bound and run under CUDA-graph replay, so the
only trustworthy attribution is to time each kernel the way the model issues
it: captured into a graph and replayed.  This sums the per-kernel times and
compares against the whole-model time, so any gap is visible rather than
assumed away.
"""
from __future__ import annotations
import gc, importlib.util, os, statistics, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT,"torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"]=bm; _spec.loader.exec_module(bm)
import triton

def graph_time(launch, reps=50, iters=20):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): launch()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): launch()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b)/reps)
    return statistics.median(ts)

def breakdown(name,B,D,H,S,L,F):
    dev=torch.device("cuda"); hd=D//H; scale=hd**-0.5; tok=B*S
    cfg=bm.TransformerConfig(B,S,D,H,F,L,True)
    m=bm.UserOptimizedTransformer(cfg).to(dev,torch.float32).eval()
    x,mask=bm.generate_random_case(cfg,dev,torch.float32,101234,0.0,1.0)
    with torch.inference_mode():
        for _ in range(10): m(x,mask)
        torch.cuda.synchronize()
        ev=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(60)]
        for a,b in ev: a.record(); m(x,mask); b.record()
        torch.cuda.synchronize()
        total=statistics.median(a.elapsed_time(b) for a,b in ev)
    pack=m._packs[0]
    h=torch.randn(tok,D,device=dev,dtype=torch.float16)
    qkv=torch.randn(tok,3*D,device=dev,dtype=torch.float16)
    ctx=torch.randn(tok,D,device=dev,dtype=torch.float16)
    hid=torch.randn(tok,F,device=dev,dtype=torch.float16)
    resid=torch.randn(tok,D,device=dev,dtype=torch.float32)
    parts={}
    if m._qkvattn:
        parts["qkv+attn (fused)"]=graph_time(lambda: bm._tl_qkvattn(
            h,pack.qkv_wt,pack.qkv_b32,B,S,H,hd,scale,torch.float16,m._qkv_cfgs))
    else:
        parts["qkv GEMM"]=graph_time(lambda: bm._tl_linear(h,pack.qkv_wt,pack.qkv_b32))
        parts["attention"]=graph_time(lambda: bm._tl_attn(
            qkv,B,S,H,hd,scale,torch.float16,m._attn_cfgs))
    parts["out proj +resid+LN"]=graph_time(lambda: bm._tl_linear_ln(
        ctx,pack.o_wt,pack.o_b32,resid,pack.n2_w,pack.n2_b))
    parts["ffn_in +GELU"]=graph_time(lambda: bm._tl_linear(h,pack.f1_wt,pack.f1_b32,gelu=True))
    parts["ffn_out +resid+LN"]=graph_time(lambda: bm._tl_linear_ln(
        hid,pack.f2_wt,pack.f2_b32,resid,pack.n1_w,pack.n1_b))
    per_layer=sum(parts.values()); acc=per_layer*L
    print(f"\n=== {name}: B={B} d={D} H={H} S={S} L={L} | model {total:.4f} ms "
          f"| graph={'Y' if m._graph is not None else 'n'} ===")
    print(f"{'kernel':>22} {'ms/call':>9} {'x L':>9} {'% model':>8}")
    for k,v in sorted(parts.items(), key=lambda kv:-kv[1]):
        print(f"{k:>22} {v:>9.5f} {v*L:>9.4f} {100*v*L/total:>7.1f}%")
    print(f"{'sum of kernels':>22} {per_layer:>9.5f} {acc:>9.4f} {100*acc/total:>7.1f}%")
    print(f"{'unattributed':>22} {'':>9} {total-acc:>9.4f} {100*(total-acc)/total:>7.1f}%")
    del m,x,mask,h,qkv,ctx,hid,resid; gc.collect(); torch.cuda.empty_cache()

for a in [("case 9  H=1",64,128,1,128,4,128),
          ("case 1  H=4",64,128,4,128,4,128),
          ("case 10 H=2",64,128,2,128,4,128)]:
    breakdown(*a)
