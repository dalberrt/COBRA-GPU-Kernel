"""Correctness checks that the shape sweep does not cover.

Development tool -- not part of the submission.

    python tools/validate.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "bm", os.path.join(ROOT, "torch_transformer_benchmark.py")
)
bm = importlib.util.module_from_spec(_spec)
sys.modules["bm"] = bm
_spec.loader.exec_module(bm)

RTOL, ATOL = 0.02, 0.002
DEV = torch.device("cuda")
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def build(cfg, dtype):
    baseline = bm.BaselineTransformer(cfg)
    optimized = bm.UserOptimizedTransformer(cfg)
    bm.copy_model_weights(baseline, optimized, strict=True)
    return (
        baseline.to(device=DEV, dtype=dtype).eval(),
        optimized.to(device=DEV, dtype=dtype).eval(),
    )


def compare(cfg, dtype, padding, trials=4, label=""):
    baseline, optimized = build(cfg, dtype)
    worst, failed = 0.0, 0
    with torch.inference_mode():
        for t in range(trials):
            x, mask = bm.generate_random_case(cfg, DEV, dtype, 1234 + t, padding, 1.0)
            r = bm.compare_outputs(
                baseline(x, mask), optimized(x, mask), rtol=RTOL, atol=ATOL
            )
            worst = max(worst, r.max_abs_error)
            failed += r.failed_elements
    check(label, failed == 0, f"failed={failed} max_abs={worst:.3e}")
    del baseline, optimized
    torch.cuda.empty_cache()


def main() -> int:
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\n[1] state_dict compatibility (strict=True load must not raise)")
    cfg = bm.TransformerConfig(4, 128, 128, 4, 128, 4, True)
    base, opt = bm.BaselineTransformer(cfg), bm.UserOptimizedTransformer(cfg)
    bk, ok_ = set(base.state_dict()), set(opt.state_dict())
    check("state_dict keys identical", bk == ok_,
          f"extra={sorted(ok_ - bk)} missing={sorted(bk - ok_)}")

    print("\n[2] padding (validates the causal + left-aligned padding argument)")
    for pad in (0.1, 0.3, 0.5, 0.9):
        compare(bm.TransformerConfig(8, 128, 128, 4, 128, 4, True), torch.float32,
                pad, label=f"causal, padding_ratio={pad}")
    for pad in (0.0, 0.3):
        compare(bm.TransformerConfig(8, 128, 128, 4, 128, 4, False), torch.float32,
                pad, label=f"NON-causal, padding_ratio={pad}")

    print("\n[3] dtypes")
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        compare(bm.TransformerConfig(64, 128, 128, 4, 128, 4, True), dt, 0.0,
                label=f"case-1 shape, dtype={str(dt).replace('torch.', '')}")
    compare(bm.TransformerConfig(64, 1024, 128, 4, 128, 4, True), torch.float16, 0.0,
            label="case-13 shape, dtype=float16")

    print("\n[4] harness default shape (non-causal, ffn=4x, 6 layers)")
    compare(bm.TransformerConfig(8, 128, 512, 8, 2048, 6, False), torch.float32, 0.0,
            label="default shape")

    print("\n[5] odd shapes")
    compare(bm.TransformerConfig(1, 1, 128, 4, 128, 4, True), torch.float32, 0.0,
            label="seq_len=1")
    compare(bm.TransformerConfig(3, 17, 96, 3, 96, 2, True), torch.float32, 0.0,
            label="non-power-of-two B=3 S=17 D=96 H=3")
    compare(bm.TransformerConfig(2, 128, 128, 4, 128, 1, True), torch.float32, 0.0,
            label="single layer")

    print(f"\n{'ALL PASSED' if not _failures else 'FAILURES: ' + ', '.join(_failures)}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
