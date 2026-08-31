"""Where does the GEMM time actually go on sm_120, and is the config list adequate?

Development tool -- not part of the submission.

`_gemm_kernel` is already wrapped in @triton.autotune keyed on (M, N, K), so
the *selection* adapts to this card by itself.  What does NOT adapt is the
candidate list `_GEMM_CONFIGS`, which was written against a 20-SM sm_86 part.
This measures the autotuned Triton GEMM against cuBLAS and against this card's
own measured fp16 peak, so we can tell whether the list is leaving anything on
the table before adding configs to it.
"""
from __future__ import annotations
import gc, importlib.util, os, statistics, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT, "torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"] = bm; _spec.loader.exec_module(bm)

def bench(fn, warmup=10, reps=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps): s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a,b in zip(s,e))

# empirical fp16 tensor-core peak on THIS card
pk = 0.0
for n in (2048, 4096, 8192):
    a = torch.randn(n,n,device="cuda",dtype=torch.float16)
    b = torch.randn(n,n,device="cuda",dtype=torch.float16)
    ms = bench(lambda: a@b)
    pk = max(pk, 2*n**3/1e12/(ms/1e3))
    del a,b; torch.cuda.empty_cache()
print(f"measured fp16 cuBLAS peak on this card: {pk:.1f} TFLOP/s\n")

# (label, M, K, N) -- the GEMMs each test case actually issues, per layer
SHAPES = [
    ("case 8  qkv",     8192, 1024, 3072),
    ("case 8  out/ffn", 8192, 1024, 1024),
    ("case 1  qkv",     8192,  128,  384),
    ("case 1  out/ffn", 8192,  128,  128),
    ("case 13 qkv",    65536,  128,  384),
    ("case 13 out/ffn",65536,  128,  128),
    ("case 6  qkv",  1280000,  128,  384),
    ("case 6  out/ffn",1280000,128,  128),
]
print(f"{'shape':>18} {'M':>8} {'K':>5} {'N':>5} {'triton ms':>10} {'cuBLAS ms':>10} "
      f"{'tri TF/s':>9} {'cub TF/s':>9} {'% peak':>7} {'chosen cfg':>34}")
for label, M, K, N in SHAPES:
    a = torch.randn(M,K,device="cuda",dtype=torch.float16)
    bt = torch.randn(K,N,device="cuda",dtype=torch.float16)
    bias = torch.randn(N,device="cuda",dtype=torch.float32)
    try:
        bm._tl_linear(a, bt, bias, out_fp32=True)          # warm autotune
        t_ms = bench(lambda: bm._tl_linear(a, bt, bias, out_fp32=True))
        best = bm._gemm_kernel.best_config
        cfg = str(best)[:34]
    except Exception as ex:
        t_ms = float("nan"); cfg = f"ERR {type(ex).__name__}"
    c_ms = bench(lambda: a@bt)
    fl = 2*M*N*K/1e12
    print(f"{label:>18} {M:>8} {K:>5} {N:>5} {t_ms:>10.4f} {c_ms:>10.4f} "
          f"{fl/(t_ms/1e3):>9.1f} {fl/(c_ms/1e3):>9.1f} {100*fl/(t_ms/1e3)/pk:>6.1f}% {cfg:>34}")
    del a,bt,bias; gc.collect(); torch.cuda.empty_cache()
