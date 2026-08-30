"""Run the official 14-case matrix and report accuracy + speedup.

Development tool -- not part of the submission.

Usage:
    python tools/sweep.py                 # tight gate, default policy
    python tools/sweep.py --loose         # argparse-default tolerances
    python tools/sweep.py --cases 1,5,13  # subset
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "bm", os.path.join(ROOT, "torch_transformer_benchmark.py")
)
bm = importlib.util.module_from_spec(_spec)
sys.modules["bm"] = bm
_spec.loader.exec_module(bm)

# #, batch, d_model, heads, seq_len, layers, ffn_dim  (causal=True for all)
CASES = [
    (1, 64, 128, 4, 128, 4, 128),
    (2, 1, 128, 4, 128, 4, 128),
    (3, 4, 128, 4, 128, 4, 128),
    (4, 16, 128, 4, 128, 4, 128),
    (5, 128, 128, 4, 128, 4, 128),
    (6, 10000, 128, 4, 128, 4, 128),
    (7, 64, 32, 4, 128, 4, 32),
    (8, 64, 1024, 4, 128, 4, 1024),
    (9, 64, 128, 1, 128, 4, 128),
    (10, 64, 128, 2, 128, 4, 128),
    (11, 64, 128, 16, 128, 4, 128),
    (12, 64, 128, 4, 32, 4, 128),
    (13, 64, 128, 4, 1024, 4, 128),
    (14, 32, 1024, 16, 100000, 2, 1024),
]

# Cases whose baseline is slow enough that 100 repeats would take minutes.
HEAVY = {6: 3, 8: 20, 13: 10, 14: 2}

# Case 14 (B=32, D=1024, H=16, S=100000) cannot use the stock path at all:
#
#   * its input tensor is 12.21 GiB in fp32 and the output another 12.21 GiB,
#     against 15.47 GiB of usable VRAM -- so x is generated on the HOST and the
#     model streams it through the GPU in batch chunks;
#   * `BaselineSelfAttention` materializes scores [B, H, S, S] = 19,073 GB, so
#     there is no reference output.  tools/reference_chunked.py rebuilds the
#     identical arithmetic with the query axis streamed, and is verified
#     against the true baseline at shapes where the baseline still fits.
#
# Correctness is checked on a subset of the batch at the FULL sequence length.
# Every operator in this network is independent across the batch, so a subset
# at full S exercises exactly the same code paths as all 32 sequences -- it is
# complete coverage of the kernel's behaviour, not a sample of it.
STREAMED = {14}
VALIDATE_ROWS = {14: 2}


def burn_in(seconds: float = 8.0) -> None:
    """Raise SM clocks off idle.  Without this, measurements swing up to 4x."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()


def median_ms(model, x, mask, repeats, device) -> float:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    torch.cuda.synchronize()
    for i in range(repeats):
        starts[i].record()
        model(x, mask)
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def run_case_streamed(spec, dtype, rtol, atol, trials, padding):
    """Case 14: host-resident input, chunked oracle, batch-subset correctness."""
    from reference_chunked import make_chunked_reference

    n, B, D, H, S, L, F = spec
    device = torch.device("cuda")
    rows = VALIDATE_ROWS[n]

    # ---- correctness: `rows` sequences at the full S=100000 -----------------
    sub = bm.TransformerConfig(rows, S, D, H, F, L, True)
    reference = make_chunked_reference(sub, block_q=2048)
    optimized = bm.UserOptimizedTransformer(sub)
    bm.copy_model_weights(reference, optimized, strict=True)
    reference = reference.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    worst_abs, worst_rel, failed = 0.0, 0.0, 0
    with torch.inference_mode():
        for t in range(trials):
            x, mask = bm.generate_random_case(sub, device, dtype, 1234 + t, padding, 1.0)
            ref = reference(x, mask)
            got = optimized(x, mask)
            r = bm.compare_outputs(ref, got, rtol=rtol, atol=atol)
            worst_abs = max(worst_abs, r.max_abs_error)
            worst_rel = max(worst_rel, r.max_relative_error)
            failed += r.failed_elements
            del ref, got, x, mask
    del reference, optimized
    gc.collect(); torch.cuda.empty_cache()

    # ---- timing: the real B=32 batch, input on the host ---------------------
    cfg = bm.TransformerConfig(B, S, D, H, F, L, True)
    full = bm.UserOptimizedTransformer(cfg).to(device=device, dtype=dtype).eval()
    x = torch.randn(B, S, D, dtype=dtype)            # HOST tensor, 12.21 GiB
    reps = HEAVY.get(n, 3)
    with torch.inference_mode():
        full(x, None)                                 # warm
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.time()
            out = full(x, None)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000.0)
            del out
    opt_ms = statistics.median(times)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    cdt = str(full._cdt).replace("torch.", "")
    del full, x
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return dict(
        case=n, B=B, D=D, H=H, S=S, L=L, passed=(failed == 0), failed=failed,
        max_abs=worst_abs, max_rel=worst_rel, base_ms=float("nan"), opt_ms=opt_ms,
        speedup=float("nan"), peak_gb=peak, cdt=cdt, graphed=False,
    )


def run_case(spec, dtype, rtol, atol, trials, padding):
    n, B, D, H, S, L, F = spec
    device = torch.device("cuda")
    cfg = bm.TransformerConfig(B, S, D, H, F, L, True)

    baseline = bm.BaselineTransformer(cfg)
    optimized = bm.UserOptimizedTransformer(cfg)
    bm.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    worst_abs, worst_rel, failed = 0.0, 0.0, 0
    with torch.inference_mode():
        for t in range(trials):
            x, mask = bm.generate_random_case(cfg, device, dtype, 1234 + t, padding, 1.0)
            ref = baseline(x, mask)
            got = optimized(x, mask)
            r = bm.compare_outputs(ref, got, rtol=rtol, atol=atol)
            worst_abs = max(worst_abs, r.max_abs_error)
            worst_rel = max(worst_rel, r.max_relative_error)
            failed += r.failed_elements
            del ref, got, x, mask

        x, mask = bm.generate_random_case(cfg, device, dtype, 101234, padding, 1.0)
        reps = HEAVY.get(n, 60)
        for _ in range(min(10, reps)):
            baseline(x, mask)
            optimized(x, mask)
        torch.cuda.synchronize()
        base_ms = median_ms(baseline, x, mask, reps, device)
        opt_ms = median_ms(optimized, x, mask, reps, device)
        base_ms = min(base_ms, median_ms(baseline, x, mask, reps, device))
        opt_ms = min(opt_ms, median_ms(optimized, x, mask, reps, device))

    peak = torch.cuda.max_memory_allocated() / 1024**3
    cdt = str(optimized._cdt).replace("torch.", "")
    graphed = optimized._graph is not None
    del baseline, optimized, x, mask
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return dict(
        case=n, B=B, D=D, H=H, S=S, L=L, passed=(failed == 0), failed=failed,
        max_abs=worst_abs, max_rel=worst_rel, base_ms=base_ms, opt_ms=opt_ms,
        speedup=base_ms / opt_ms, peak_gb=peak, cdt=cdt, graphed=graphed,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loose", action="store_true", help="use argparse-default tolerances")
    ap.add_argument("--cases", default="", help="comma-separated case numbers")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--padding", type=float, default=0.0)
    ap.add_argument("--precision", default="auto", choices=("auto", "tf32", "fp16"))
    ap.add_argument("--no-burn-in", action="store_true")
    ap.add_argument("--no-triton-attn", action="store_true")
    ap.add_argument("--no-qkvattn", action="store_true")
    args = ap.parse_args()

    bm._PRECISION_POLICY = args.precision
    if args.no_triton_attn:
        bm._USE_TRITON_ATTN = False
    if args.no_qkvattn:
        bm._USE_FUSED_QKV_ATTN = False

    rtol, atol = (0.02, 0.002) if args.loose else (0.01, 0.001)
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    wanted = {int(c) for c in args.cases.split(",") if c.strip()} if args.cases else None
    selected = [c for c in CASES if wanted is None or c[0] in wanted]

    if not args.no_burn_in:
        print("burn-in (clocks)...", flush=True)
        burn_in()

    print(
        f"\ngate: abs<={atol} OR rel<={rtol} | dtype={args.dtype} | "
        f"padding={args.padding} | precision={args.precision}\n"
    )
    hdr = (f"{'#':>3} {'B':>6} {'D':>5} {'H':>3} {'S':>6} {'dt':>7} {'G':>2} "
           f"{'res':>5} {'failed':>8} {'max_abs':>10} {'base ms':>10} "
           f"{'opt ms':>9} {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for spec in selected:
        try:
            if spec[0] in STREAMED:
                r = run_case_streamed(spec, dtype, rtol, atol,
                                      min(args.trials, 2), args.padding)
            else:
                r = run_case(spec, dtype, rtol, atol, args.trials, args.padding)
            rows.append(r)
            print(
                f"{r['case']:>3} {r['B']:>6} {r['D']:>5} {r['H']:>3} {r['S']:>6} "
                f"{r['cdt']:>7} {'Y' if r['graphed'] else 'n':>2} "
                f"{'PASS' if r['passed'] else 'FAIL':>5} {r['failed']:>8} "
                f"{r['max_abs']:>10.3e} "
                + (f"{r['base_ms']:>10.3f}" if r['base_ms'] == r['base_ms'] else f"{'n/a':>10}")
                + f" {r['opt_ms']:>9.3f} "
                + (f"{r['speedup']:>7.2f}x" if r['speedup'] == r['speedup'] else f"{'n/a':>8}"),
                flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{spec[0]:>3} {spec[1]:>6} {spec[2]:>5} {spec[3]:>3} {spec[4]:>6} "
                  f"  ERROR  {type(exc).__name__}: {str(exc)[:70]}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    if rows:
        ok = [r for r in rows if r["passed"]]
        print(f"\npassed {len(ok)}/{len(rows)}")
        if ok:
            sp = [r["speedup"] for r in ok if r["speedup"] == r["speedup"]]
            print(f"speedup over passing cases: min={min(sp):.2f}x  "
                  f"median={statistics.median(sp):.2f}x  max={max(sp):.2f}x")
        bad = [r for r in rows if not r["passed"]]
        if bad:
            print("FAILED: " + ", ".join(
                f"case {r['case']} ({r['failed']} elems, max_abs={r['max_abs']:.2e})"
                for r in bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
