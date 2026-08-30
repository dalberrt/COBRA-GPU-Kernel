"""Re-derive the CUDA-graph capture gate for THIS driver model.

Development tool -- not part of the submission.

`_prepare` gates capture on a fixed budget:

    self._graph_ok = x.is_cuda and x.numel() * x.element_size() <= 64 MiB

That constant was chosen on an RTX 3050 under **Windows/WDDM**, where the
friend measured the baseline as "CPU-DISPATCH-bound, not GPU-bound" and found
graph capture to be "the first-order win".  WDDM batches submissions through a
kernel-mode scheduler and its per-launch cost is far higher than Linux's.  On
Linux a launch is single-digit microseconds, so the gate has to be re-measured
rather than inherited.

This A/Bs every case with capture forced on and forced off.
"""
from __future__ import annotations
import argparse, gc, importlib.util, os, statistics, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT,"torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"]=bm; _spec.loader.exec_module(bm)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import CASES, HEAVY  # noqa: E402

def median_ms(model, x, mask, reps):
    s=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    torch.cuda.synchronize()
    for i in range(reps): s[i].record(); model(x,mask); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a,b in zip(s,e))

def run(spec, force_graph):
    n,B,D,H,S,L,F = spec
    dev=torch.device("cuda")
    cfg=bm.TransformerConfig(B,S,D,H,F,L,True)
    m=bm.UserOptimizedTransformer(cfg).to(dev,torch.float32).eval()
    x,mask=bm.generate_random_case(cfg,dev,torch.float32,101234,0.0,1.0)
    with torch.inference_mode():
        m(x,mask)                      # triggers _prepare
        if force_graph is not None:
            m._graph_ok = force_graph
            m._graph = None
            m._static_x = m._static_m = m._static_out = None
        for _ in range(10): m(x,mask)
        torch.cuda.synchronize()
        reps = HEAVY.get(n, 60)
        ms = min(median_ms(m,x,mask,reps), median_ms(m,x,mask,reps))
        graphed = m._graph is not None
    xb = x.numel()*x.element_size()
    del m,x,mask; gc.collect(); torch.cuda.empty_cache()
    return ms, graphed, xb

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--cases",default="1,2,3,4,5,6,7,8,9,10,11,12,13")
    a=ap.parse_args()
    torch.manual_seed(1234); torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32=True
    print("burn-in..."); 
    _a=torch.randn(2048,2048,device="cuda",dtype=torch.float16)
    import time; t=time.time()
    while time.time()-t<8:
        for _ in range(20): _a@_a
        torch.cuda.synchronize()
    del _a; torch.cuda.empty_cache()
    want={int(c) for c in a.cases.split(",") if c.strip()}
    print(f"\n{'#':>3} {'B':>6} {'S':>6} {'x bytes':>10} {'shipped':>8} "
          f"{'graph ms':>9} {'eager ms':>9} {'graph win':>10}  verdict")
    for spec in CASES:
        if spec[0] not in want: continue
        try:
            g_ms,g_on,xb = run(spec, True)
            e_ms,_,_     = run(spec, False)
        except Exception as ex:
            print(f"{spec[0]:>3}  ERROR {type(ex).__name__}: {str(ex)[:50]}"); continue
        shipped = "graph" if xb <= 64*1024*1024 else "eager"
        win = e_ms/g_ms
        verdict = "graph" if win > 1.02 else ("eager" if win < 0.98 else "tie")
        print(f"{spec[0]:>3} {spec[1]:>6} {spec[4]:>6} {xb/2**20:>9.1f}M {shipped:>8} "
              f"{g_ms:>9.4f} {e_ms:>9.4f} {win:>9.3f}x  {verdict}"
              + ("" if verdict==shipped or verdict=="tie" else "   <-- gate disagrees"))
