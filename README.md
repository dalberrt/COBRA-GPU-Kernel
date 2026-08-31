# Blackwell Transformer Kernel — sm_120

A fused Triton implementation of the TikTok TechJam Transformer-layer benchmark,
tuned for and measured on an **NVIDIA RTX 5060 Ti (Blackwell, sm_120, 16 GB)**.

**Median 9.19x** over the 13 test shapes that have a runnable reference.
**All 14 shapes execute correctly, with zero failing elements.**

Four Triton kernels per layer replace the reference's ~29 PyTorch dispatches,
the forward runs as a single CUDA-graph replay wherever that pays, and shapes
whose activations exceed VRAM are streamed in batch chunks sized from the
device's actual free memory.

## Results

| # | geometry | speedup | | # | geometry | speedup |
|---|---|---|---|---|---|---|
| 1 | B64 d128 H4 S128 | 5.19x | | 8 | B64 d1024 H4 S128 | 2.87x |
| 2 | B1 d128 H4 S128 | 16.11x | | 9 | B64 d128 H1 S128 | 2.97x |
| 3 | B4 d128 H4 S128 | 11.61x | | 10 | B64 d128 H2 S128 | 4.07x |
| 4 | B16 d128 H4 S128 | 7.81x | | 11 | B64 d128 H16 S128 | 25.33x |
| 5 | B128 d128 H4 S128 | 9.64x | | 12 | B64 d128 H4 S32 | 8.09x |
| 6 | B10000 d128 H4 S128 | 9.19x | | 13 | B64 d128 H4 S1024 | 33.34x |
| 7 | B64 d32 H4 S128 | 9.78x | | **14** | **B32 d1024 H16 S100000** | **runs — see below** |

| statistic | value |
|---|---|
| median (13 shapes) | **9.19x** |
| geometric mean | 8.67x |
| arithmetic mean | 11.23x |
| worst `max_abs` | 1.35e-03 against a 2e-3 limit |
| failing elements | **0**, every shape |

Gate: `abs_err <= 0.002` **OR** `rel_err <= 0.02`, per element.

### Shape 14, which is the interesting one

B=32, d=1024, H=16, **S=100000**. The reference cannot execute it on any
hardware — `BaselineSelfAttention` materializes `scores[B,H,S,S]`, which is
**19,073 GB** — and the input/output pair alone is 24.41 GiB against 15.47 GiB
of VRAM. It therefore has no speedup ratio, and none is claimed.

This implementation runs it in **37.67 s at 36.9 TFLOP/s — 73% of this card's
measured fp16 peak** — with **0 failing elements** (`max_abs` 7.14e-04),
validated against a bit-exact oracle at the full S=100000. Per-sequence cost is
**1.14–1.18 s across a 16x batch range**, so the streaming decomposition adds no
measurable per-chunk overhead.

## What this branch contributes

Roughly 1900 lines across the kernel policy layer, a measurement suite, and a
correctness oracle:

| area | contribution |
|---|---|
| **Shape 14** | Host-streamed batch chunking sized from `mem_get_info()` with halve-on-OOM, plus `tools/reference_chunked.py` — a memory-safe oracle verified `torch.equal`-identical to the stock baseline across 16 configurations. Takes shape 14 from *unrunnable* to *correct in 37.67 s*. |
| **Precision policy** | Found the `float32` floor below `d_model=64` was silently disabling the Triton GEMM and falling back to cuBLAS TF32 — same 10 mantissa bits as fp16, without the fp32-store correction. Removing it: **1.34x faster and more accurate** (360-trial campaign). |
| **Fusion gate** | The QKV+attention fusion needs `BM >= S`, which ties the projection's tile height to sequence length. Added a lower `seq_len` bound after measuring it 1.13x *slower* at S=32. |
| **Attention tiles** | Re-derived against 36 SMs; `head_dim=64` tile 1.22x, `head_dim` gate raised 128 -> 256 after measuring the Triton kernel 1.11x faster than FlashAttention-2 there. |
| **Measurement suite** | 12 standalone harnesses (`tools/`) — tile search with an analytic smem/occupancy pruner, in-graph timing, SDPA backend comparison, graph-gate A/B, precision campaigns, per-kernel breakdown. |
| **Analysis** | `report/measurements/` — raw logs plus write-ups, including nine approaches that measurement killed. |

## Relationship to `main`

This branch builds on the **COBRA** GPU kernel project by another author
(preserved unmodified on `main`, with its own report in
`report/TECH_REPORT.md`). That work contributed the fused megakernel structure,
the fp32-store GEMM, the CUDA-graph capture and the LayerNorm-in-epilogue
fusion, tuned on an RTX 3050 (sm_86, 20 SMs) under Windows/WDDM.

The five Triton kernel bodies are inherited unchanged. This branch re-derives
every hardware-specific constant against measurement on sm_120, corrects two
policy decisions that do not transfer, adds the shape-14 capability, and builds
the measurement infrastructure that supports all of it.

```bash
git diff main...port/blackwell-sm120     # exactly what this branch changes
```

## Hardware

| | value |
|---|---|
| GPU | RTX 5060 Ti, 16 GB (15.47 GiB usable), sm_120 |
| SMs | 36 |
| Shared memory | 101376 B/block opt-in, 102400 B/SM |
| Registers | 65536 per SM |
| L2 | 32 MB |
| **Measured fp16 peak** | **50.3 TFLOP/s** (cuBLAS, square GEMMs) |
| **Measured bandwidth** | **387 GB/s** (copy, read+write) |
| CPU / RAM | AMD Ryzen 5 7600 (6c/12t) / 29 GiB |
| Software | Driver 595.84, PyTorch 2.11.0+cu128, Triton 3.6.0, Python 3.14.4, Linux |

Consumer Blackwell halves fp16-with-fp32-accumulate throughput, so achievable
peak is far below the marketing figure. Every efficiency claim here is against
the measured 50.3 TFLOP/s.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
# sm_120 needs PyTorch 2.11+ and Triton 3.6+; Triton ships with the wheel.
# No compiler toolchain required — every kernel is JIT-compiled at first use.
```

## Reproducing

```bash
./.venv/bin/python tools/sweep.py --loose          # the 14-case matrix
./.venv/bin/python tools/validate.py               # coverage the sweep misses
cd tools && ../.venv/bin/python reference_chunked.py   # prove the oracle is bit-exact

# individual findings
./.venv/bin/python tools/graph_gate.py     # the ~0.42 ms Linux dispatch floor
./.venv/bin/python tools/sdpa_backend.py   # Triton vs FlashAttention-2
./.venv/bin/python tools/diag_small.py     # in-graph vs wrapper timing
./.venv/bin/python tools/breakdown.py      # per-kernel attribution
./.venv/bin/python tools/time_case14.py    # shape-14 wall clock
```

`sweep.py` runs an 8-second clock burn-in first — SM clocks off idle swing
results by up to 4x on this card. Shape 14 needs ~25 GiB of host RAM and takes
~40 s per forward.

## Limitations

* **Speedup has an unstable denominator.** The reference's own timings swing up
  to 2.63x run-to-run on cases 4, 5 and 7; this implementation's times hold to
  within 1.4%. Every figure is corroborated by at least two independent sweeps.
* **No kernel-level profiler.** CUPTI fails with `CUPTI_ERROR_INVALID_DEVICE`
  and Nsight needs root (`RmProfilingAdminOnly: 1`), so occupancy is *modelled*,
  never measured — which is why every model-ranked config was re-confirmed by
  direct timing before adoption.
* **Shape 14 needs a host-resident input.** If the harness allocates x on the
  GPU, 12.21 GiB is gone before this code runs. A property of the harness and a
  16 GB card, not of the kernel.
* **Case 6 carries a pre-change measurement** and sits exactly at the median of
  13; re-measurement was blocked by an unrelated process holding VRAM.
* **The fp32 store leaves up to 2.39x unclaimed** on bandwidth-bound shapes, but
  the accuracy margin under input-scale stress is too thin to spend it.

Full analysis, including the nine rejected approaches, is in
`report/measurements/`.
