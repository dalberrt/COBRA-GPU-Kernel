"""Why is the thin-K Triton GEMM behind cuBLAS on sm_120?

Development tool -- not part of the submission.

tune_gemm.py showed the autotuned Triton GEMM at 96-97% of this card's
measured cuBLAS peak on the wide-K case-8 shapes, but well behind on the
thin-K (K=128) shapes every small test case issues.  Rather than brute-force
the config space (which is compile-bound and was going nowhere), this isolates
the candidate causes:

  (a) the fp32 output store, adopted for accuracy on sm_86, which writes 2x
      the bytes of cuBLAS's fp16 store;
  (b) the config list itself;
  (c) the fused epilogue (bias / GELU / residual) cuBLAS is not performing.

If (a) dominates, these GEMMs are bandwidth-bound and no config will help.
"""
from __future__ import annotations
import gc, importlib.util, os, statistics, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT,"torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"]=bm; _spec.loader.exec_module(bm)

def bench(fn, warmup=10, reps=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps): s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a,b in zip(s,e))

def main():
    # measured achievable bandwidth on this card, not the spec sheet
    n = 1 << 26
    src = torch.empty(n, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    ms = bench(lambda: dst.copy_(src))
    bw = 2 * n * 2 / (ms/1e3)          # read + write
    print(f"measured copy bandwidth on this card: {bw/1e9:.0f} GB/s\n")
    del src, dst; torch.cuda.empty_cache()

    SHAPES=[("case 1/4/5/9 out+ffn",8192,128,128),
            ("case 1/4/5/9 qkv",    8192,128,384),
            ("case 13 out+ffn",    65536,128,128),
            ("case 6 qkv",       1280000,128,384)]
    print(f"{'shape':>22} {'tri fp32':>9} {'tri fp16':>9} {'cuBLAS':>9} "
          f"{'fp32 store costs':>17} {'tri16 vs cub':>13} {'fp32 %BW':>9}")
    for label,M,K,N in SHAPES:
        a=torch.randn(M,K,device="cuda",dtype=torch.float16)
        bt=torch.randn(K,N,device="cuda",dtype=torch.float16)
        bias=torch.randn(N,device="cuda",dtype=torch.float32)
        bm._tl_linear(a,bt,bias,out_fp32=True)
        t32=bench(lambda: bm._tl_linear(a,bt,bias,out_fp32=True))
        bm._tl_linear(a,bt,bias,out_fp32=False)
        t16=bench(lambda: bm._tl_linear(a,bt,bias,out_fp32=False))
        cub=bench(lambda: a@bt)
        by = (M*K + K*N)*2 + M*N*4       # bytes the fp32-store variant moves
        print(f"{label:>22} {t32:>9.4f} {t16:>9.4f} {cub:>9.4f} "
              f"{t32/t16:>16.2f}x {t16/cub:>12.2f}x {100*by/(t32/1e3)/bw:>8.0f}%")
        del a,bt,bias; gc.collect(); torch.cuda.empty_cache()

main()
