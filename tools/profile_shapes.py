"""Per-kernel CUDA profile of the optimized model, per shape.

Development tool -- not part of the submission.  Answers "where does the
time actually go on THIS card" before any constant is re-derived.

    python tools/profile_shapes.py --cases 1,8,9,10
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "bm", os.path.join(ROOT, "torch_transformer_benchmark.py")
)
bm = importlib.util.module_from_spec(_spec)
sys.modules["bm"] = bm
_spec.loader.exec_module(bm)

from sweep import CASES  # noqa: E402  (same dir)


def profile_case(spec, topk: int) -> None:
    n, B, D, H, S, L, F = spec
    dev = torch.device("cuda")
    cfg = bm.TransformerConfig(B, S, D, H, F, L, True)
    model = bm.UserOptimizedTransformer(cfg).to(device=dev, dtype=torch.float32).eval()
    x, mask = bm.generate_random_case(cfg, dev, torch.float32, 101234, 0.0, 1.0)

    with torch.inference_mode():
        for _ in range(20):           # warm autotune + graph capture
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(20):
                model(x, mask)
            torch.cuda.synchronize()

    evts = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in evts)
    print(f"\n=== case {n}: B={B} D={D} H={H} S={S} L={L} | "
          f"cdt={str(model._cdt).replace('torch.','')} "
          f"graph={'Y' if model._graph is not None else 'n'} "
          f"| {total/20/1000:.3f} ms/iter ===")
    print(f"{'kernel':<58} {'ms/iter':>9} {'%':>6}")
    for e in sorted(evts, key=lambda e: -e.self_device_time_total)[:topk]:
        print(f"{e.key[:58]:<58} {e.self_device_time_total/20/1000:>9.4f} "
              f"{100*e.self_device_time_total/total:>5.1f}%")

    del model, x, mask
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,8,9,10")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    wanted = {int(c) for c in args.cases.split(",") if c.strip()}
    for spec in CASES:
        if spec[0] in wanted:
            profile_case(spec, args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
