"""True in-graph GPU time of the small GEMMs.

Development tool -- not part of the submission.

diag_gemm.py measured 0.0299 ms for M=8192 regardless of store dtype and at
only 55% of bandwidth -- the signature of host-side overhead (the Python
wrapper plus the autotuner's cache lookup), not of GPU work.  The model runs
these inside a CUDA graph where that overhead does not exist, so timing them
through the Python wrapper overstates their cost.

This captures N launches into a CUDA graph and replays it, which is exactly
how `_run_graphed` executes them, and compares against cuBLAS captured the
same way.
"""
from __future__ import annotations
import gc, importlib.util, os, statistics, sys
import torch, triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT,"torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"]=bm; _spec.loader.exec_module(bm)

REPS = 50

def graph_time(launch, reps=REPS, iters=20):
    """Median ms for ONE launch, measured inside a CUDA graph."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): launch()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
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

SHAPES=[("case 1/4/5/9 out+ffn",8192,128,128),
        ("case 1/4/5/9 qkv",    8192,128,384),
        ("case 13 out+ffn",    65536,128,128),
        ("case 7 out+ffn",      8192, 32,  32)]

print(f"{'shape':>22} {'M':>7} {'K':>4} {'N':>4} {'triton fp32':>12} "
      f"{'cuBLAS fp16':>12} {'ratio':>7}  chosen config")
for label,M,K,N in SHAPES:
    a=torch.randn(M,K,device="cuda",dtype=torch.float16)
    bt=torch.randn(K,N,device="cuda",dtype=torch.float16)
    bias=torch.randn(N,device="cuda",dtype=torch.float32)
    o32=torch.empty((M,N),device="cuda",dtype=torch.float32)
    o16=torch.empty((M,N),device="cuda",dtype=torch.float16)

    bm._tl_linear(a,bt,bias,out_fp32=True)              # resolve autotune
    cfg = bm._gemm_kernel.best_config
    BM,BN,BK,GM = cfg.kwargs["BM"],cfg.kwargs["BN"],cfg.kwargs["BK"],cfg.kwargs["GROUP_M"]
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    def tri():
        bm._gemm_kernel.fn[grid](a,bt,bias,o32,o32,o32,M,N,K,
            a.stride(0),a.stride(1),bt.stride(0),bt.stride(1),o32.stride(0),o32.stride(1),
            GELU=False,OUT_FP32=True,ADD_RESID=False,MASK_RESID=False,
            BM=BM,BN=BN,BK=BK,GROUP_M=GM,
            num_warps=cfg.num_warps,num_stages=cfg.num_stages)
    t = graph_time(tri)
    c = graph_time(lambda: torch.mm(a,bt,out=o16))
    print(f"{label:>22} {M:>7} {K:>4} {N:>4} {t:>12.5f} {c:>12.5f} {t/c:>6.2f}x  "
          f"BM={BM} BN={BN} BK={BK} w={cfg.num_warps} s={cfg.num_stages}")
    del a,bt,bias,o32,o16; gc.collect(); torch.cuda.empty_cache()
