"""Re-examine the precision policy for Blackwell tensor cores.

Development tool -- not part of the submission.

`_prepare` picks `float32 if d_model <= 64 else float16`.  The friend's stated
reason: "at d_model=32 the fp16 tail reached 1.92e-3 over 24 trials against a
2e-3 gate, while TF32 stays at 6.6e-4.  Not worth 2x."  That was measured on
sm_86.  This re-measures the tail on sm_120 over many more trials before
deciding whether the fp32 floor is still needed here.

Accuracy only -- no timing -- so it is safe to run alongside other GPU work.
"""
from __future__ import annotations
import argparse, gc, importlib.util, os, statistics, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bm", os.path.join(ROOT,"torch_transformer_benchmark.py"))
bm = importlib.util.module_from_spec(_spec); sys.modules["bm"]=bm; _spec.loader.exec_module(bm)

ATOL, RTOL = 0.002, 0.02

def campaign(B,D,H,S,L,F, policy, trials, scales=(1.0,), pads=(0.0,)):
    dev=torch.device("cuda")
    bm._PRECISION_POLICY = policy
    cfg=bm.TransformerConfig(B,S,D,H,F,L,True)
    base=bm.BaselineTransformer(cfg); opt=bm.UserOptimizedTransformer(cfg)
    bm.copy_model_weights(base,opt,strict=True)
    base=base.to(dev,torch.float32).eval(); opt=opt.to(dev,torch.float32).eval()
    worst=0.0; failed=0; n=0; cdt=None
    with torch.inference_mode():
        for sc in scales:
            for pad in pads:
                for t in range(trials):
                    x,mask=bm.generate_random_case(cfg,dev,torch.float32,1000+t,pad,sc)
                    r=bm.compare_outputs(base(x,mask),opt(x,mask),rtol=RTOL,atol=ATOL)
                    worst=max(worst,r.max_abs_error); failed+=r.failed_elements; n+=1
                    cdt = str(opt._cdt).replace("torch.","")
                    del x,mask
    del base,opt; gc.collect(); torch.cuda.empty_cache()
    return cdt, worst, failed, n

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--trials",type=int,default=30)
    a=ap.parse_args()
    torch.manual_seed(1234); torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32=True
    SHAPES=[("case 7  D=32", 64,32,4,128,4,32),
            ("case 1  D=128",64,128,4,128,4,128),
            ("D=64 (gate edge)",64,64,4,128,4,64)]
    print(f"gate: abs<={ATOL} OR rel<={RTOL} | {a.trials} trials x 3 scales x 2 paddings\n")
    print(f"{'shape':>18} {'policy':>7} {'cdt':>8} {'worst max_abs':>14} {'margin':>8} {'failed':>7} {'n':>4}")
    for label,B,D,H,S,L,F in SHAPES:
        for pol in ("auto","fp16"):
            cdt,worst,failed,n = campaign(B,D,H,S,L,F,pol,a.trials,
                                          scales=(0.5,1.0,2.0), pads=(0.0,0.3))
            print(f"{label:>18} {pol:>7} {cdt:>8} {worst:>14.4e} "
                  f"{ATOL/worst:>7.2f}x {failed:>7} {n:>4}")
