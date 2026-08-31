"""Wall-clock for shape 14, logged incrementally so a partial run still yields data.

Development tool -- not part of the submission.  Earlier attempts to time this
inside the full sweep were lost to session teardown mid-measurement, so this
does the timing alone, prints after every step, and scales the batch upward so
a failure at B=32 still leaves usable per-sequence numbers behind.
"""
from __future__ import annotations
import gc, importlib.util, os, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT,"torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"]=bm; _spec.loader.exec_module(bm)

S,D,H,L,F = 100000,1024,16,2,1024
dev=torch.device("cuda")
TFLOP_per_seq = (2*(2*S*S*(D//H)/2)*H*L + (4*2*S*D*D + 2*2*S*D*F)*L)/1e12

print(f"shape 14: S={S} d={D} H={H} L={L} | {TFLOP_per_seq:.1f} TFLOP per sequence", flush=True)
print(f"{'B':>4} {'host GiB':>9} {'reps':>5} {'median s':>10} {'s / seq':>9} {'TFLOP/s':>9} {'chunk':>6}", flush=True)

for B in (2, 4, 8, 32):
    try:
        cfg = bm.TransformerConfig(B,S,D,H,F,L,True)
        m = bm.UserOptimizedTransformer(cfg).to(dev, torch.float32).eval()
        x = torch.randn(B,S,D, dtype=torch.float32)          # HOST tensor
        gib = 2*x.numel()*4/2**30
        reps = 3 if B <= 8 else 2
        with torch.inference_mode():
            out = m(x, None); torch.cuda.synchronize()
            step = m._stream_chunk_size(x)
            del out
            ts=[]
            for _ in range(reps):
                t0=time.time(); out=m(x,None); torch.cuda.synchronize()
                ts.append(time.time()-t0); del out
        ts.sort(); med=ts[len(ts)//2]
        print(f"{B:>4} {gib:>9.2f} {reps:>5} {med:>10.2f} {med/B:>9.2f} "
              f"{B*TFLOP_per_seq/med:>9.1f} {step:>6}", flush=True)
        del m,x; gc.collect(); torch.cuda.empty_cache()
    except Exception as ex:
        print(f"{B:>4}  FAILED  {type(ex).__name__}: {str(ex)[:80]}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        break
