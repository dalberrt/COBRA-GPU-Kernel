"""Re-derive the Triton attention tile configs for THIS GPU.

Development tool -- not part of the submission.

The shipped ladders in `_attn_cfgs` / `_qkvattn_cfgs` were hand-picked "by
measured register pressure on sm_86" -- a 20-SM RTX 3050 under Windows/WDDM.
This card is sm_120: 36 SMs, 32 MB L2, Linux.  Both the occupancy target and
the wave quantization differ, so the ladder has to be re-measured.

Blind grid search is the wrong tool here: Triton *compilation* dominates
(the GPU sits at 0% while 256 configs build).  So candidates are first pruned
with an explicit shared-memory and register model -- per the
`triton.optimize-triton-block-parameters` and
`patterns.choose-tile-size-and-work-partitioning` playbooks -- and only the
survivors are compiled and timed.

    python tools/tune_attn.py --cases 9,13
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import os
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "bm", os.path.join(ROOT, "torch_transformer_benchmark.py")
)
bm = importlib.util.module_from_spec(_spec)
sys.modules["bm"] = bm
_spec.loader.exec_module(bm)
import triton  # noqa: E402

_P = torch.cuda.get_device_properties(0)
SMS = _P.multi_processor_count
SMEM_BLOCK = _P.shared_memory_per_block_optin
SMEM_SM = _P.shared_memory_per_multiprocessor
REGS_SM = _P.regs_per_multiprocessor

# case -> (B, D, H, S)
CASES = {
    1: (64, 128, 4, 128), 2: (1, 128, 4, 128), 5: (128, 128, 4, 128),
    6: (10000, 128, 4, 128), 7: (64, 32, 4, 128), 8: (64, 1024, 4, 128),
    9: (64, 128, 1, 128), 10: (64, 128, 2, 128), 11: (64, 128, 16, 128),
    12: (64, 128, 4, 32), 13: (64, 128, 4, 1024),
}


def model_costs(BM, BN, hd_pad, nw, ns):
    """Static smem / register estimate, before paying a compile."""
    # Live tiles in the KV loop: Q (persistent) + K,V (pipelined ns deep).
    smem = (BM * hd_pad + ns * 2 * BN * hd_pad) * 2
    # fp32 accumulator + fp32 score tile, spread over the block's threads.
    regs = (BM * hd_pad + BM * BN) / (32.0 * nw)
    return smem, regs


def occupancy(BM, BN, hd_pad, nw, ns):
    """Blocks/SM and achieved occupancy under the smem+register model."""
    smem, regs = model_costs(BM, BN, hd_pad, nw, ns)
    threads = 32 * nw
    by_smem = SMEM_SM // max(1, smem)
    by_regs = REGS_SM // max(1, int(regs) * threads) if regs * threads > 0 else 32
    by_thr = _P.max_threads_per_multi_processor // threads
    blocks = max(0, min(by_smem, by_regs, by_thr, 32))
    occ = blocks * threads / _P.max_threads_per_multi_processor
    return smem, regs, blocks, min(1.0, occ)


def candidates(S, hd_pad, B=1, H=1, cap=36, max_regs=200.0):
    """Architecturally valid configs, ranked, capped.

    Rejected on paper rather than at a 3-second compile:
      * smem over the per-block opt-in limit  -> will not launch
      * modelled registers over ~200/thread   -> spills to local memory
      * fewer than 4 tile elements per thread -> warp lanes idle

    Survivors are ranked by (achieved occupancy x wave-quantization
    efficiency).  Wave efficiency is the sm_120-specific term: with 36 SMs a
    grid of 64 CTAs runs 1.78 waves, i.e. the second wave is 44% empty.
    """
    S2 = max(16, triton.next_power_of_2(S))
    scored = []
    for BM, BN, nw, ns in itertools.product(
        [b for b in (16, 32, 64, 128, 256) if b <= S2],
        [b for b in (16, 32, 64, 128, 256) if b <= S2],
        (1, 2, 4, 8), (1, 2, 3, 4),
    ):
        smem, regs, blocks, occ = occupancy(BM, BN, hd_pad, nw, ns)
        if smem > SMEM_BLOCK or regs > max_regs or blocks < 1:
            continue
        if BM * BN < 32 * nw * 4:
            continue
        ctas = triton.cdiv(S, BM) * H * B
        waves = ctas / SMS
        quant = waves / max(1e-9, -(-ctas // SMS) * 1.0) * (SMS / SMS)
        quant = waves / max(1.0, float(int(waves) + (waves % 1 > 0)))
        scored.append((occ * quant, BM, BN, nw, ns, smem, regs))
    scored.sort(reverse=True)
    return [(BM, BN, nw, ns, smem, regs)
            for _, BM, BN, nw, ns, smem, regs in scored[:cap]]


def bench(fn, warmup=10, reps=30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in zip(s, e))


def reference_ctx(qkv, B, S, H, hd, scale):
    q, k, v = qkv.view(B, S, 3, H, hd).permute(2, 0, 3, 1, 4).unbind(0)
    return (F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                           is_causal=True, scale=scale)
            .transpose(1, 2).reshape(B * S, H * hd))


def sweep_attn(n, B, D, H, S, topk):
    hd = D // H
    hd_pad = max(16, triton.next_power_of_2(hd))
    scale = hd ** -0.5
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    qkv = torch.randn(B * S, 3 * D, device=dev, dtype=torch.float16) * 0.5
    ref = reference_ctx(qkv, B, S, H, hd, scale)

    shipped = bm._attn_cfgs(S, hd)[0]
    cands = candidates(S, hd_pad, B, H)
    print(f"\n=== case {n}: B={B} D={D} H={H} S={S} hd={hd} | _tl_attn ===")
    print(f"{SMS} SMs | smem/block {SMEM_BLOCK} B | shipped(sm_86) "
          f"BM={shipped[0]} BN={shipped[1]} w={shipped[2]} s={shipped[3]}")
    print(f"measuring top {len(cands)} of 400 by (occupancy x wave-efficiency)")
    print(f"{'ms':>9} {'BM':>4} {'BN':>4} {'w':>2} {'s':>2} {'smemKB':>7} "
          f"{'regs/t':>7} {'waves':>6} {'max_abs':>10}", flush=True)

    results = []
    for BM, BN, nw, ns, smem, regs in cands:
        try:
            out = bm._tl_attn(qkv, B, S, H, hd, scale, torch.float16,
                              [(BM, BN, nw, ns)])
            err = (out.float() - ref).abs().max().item()
            if err > 0.02:
                continue
            ms = bench(lambda: bm._tl_attn(qkv, B, S, H, hd, scale,
                                           torch.float16, [(BM, BN, nw, ns)]))
        except Exception:
            continue
        waves = triton.cdiv(S, BM) * H * B / SMS
        results.append((ms, BM, BN, nw, ns, smem, regs, waves, err))
        print(f"{ms:>9.4f} {BM:>4} {BN:>4} {nw:>2} {ns:>2} {smem/1024:>7.1f} "
              f"{regs:>7.1f} {waves:>6.2f} {err:>10.2e}", flush=True)

    if not results:
        print("no valid config")
        return
    results.sort()
    ship_ms = next((r[0] for r in results
                    if (r[1], r[2], r[3], r[4]) == shipped), None)
    print(f"\n  BEST case {n}: BM={results[0][1]} BN={results[0][2]} "
          f"w={results[0][3]} s={results[0][4]} -> {results[0][0]:.4f} ms")
    if ship_ms:
        print(f"  shipped sm_86 config -> {ship_ms:.4f} ms  "
              f"| gain {ship_ms/results[0][0]:.3f}x")
    else:
        print("  shipped sm_86 config was rejected by the model or failed")
    del qkv, ref
    gc.collect(); torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="9,13")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(1234)
    for c in args.cases.split(","):
        if c.strip():
            B, D, H, S = CASES[int(c)]
            sweep_attn(int(c), B, D, H, S, args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
