# COBRA-GPU-Kernel — Blackwell (sm_120) port

A fused GPU implementation of the TikTok TechJam Transformer-layer benchmark,
**ported and re-tuned from [COBRA](https://github.com/dalberrt/COBRA-GPU-Kernel)**
(an sm_86 submission by a different author) to an NVIDIA RTX 5060 Ti.

## What this repository is, honestly

The kernel *design* is not mine. The fused Triton flash-attention megakernel,
the fp32-store GEMM that makes fp16 safe, the CUDA-graph capture, and the
LayerNorm-in-epilogue fusion are all from the original COBRA project, whose
author tuned them on an **RTX 3050 (sm_86, 20 SMs) running Windows/WDDM**.

What is mine is the **port**: re-deriving every hardware-specific constant
against a *measurement on this card* instead of inheriting it, and making the
one test shape that the original could not run actually run.

`main` is the unmodified original and is kept as the before/after baseline.
All of this work is on `port/blackwell-sm120`, so `git diff main` is exactly
the set of changes attributable to this port.

The substance of this submission is the distinction between:

* **what transferred** — and is therefore left alone, with the measurement
  showing why changing it would be churn; and
* **what did not transfer** — and is re-derived, with before/after numbers.

Reporting the first category is as much a part of the work as the second.

## Hardware this is tuned for

| | Original target (sm_86) | **This port (sm_120)** |
|---|---|---|
| GPU | RTX 3050, 8 GB | **RTX 5060 Ti, 16 GB** |
| SMs | 20 | **36** |
| Shared memory / block (opt-in) | 101376 B | 101376 B |
| Registers / SM | 65536 | 65536 |
| L2 cache | ~2 MB | **32 MB** |
| Driver model | **Windows / WDDM** | **Linux** |
| Measured fp16 peak | — | **50.3 TFLOP/s** |
| Measured copy bandwidth | — | **387 GB/s** |

CPU: AMD Ryzen 5 7600 (6c/12t), 29 GiB RAM.
Software: Driver 595.84, PyTorch 2.11.0+cu128, Triton 3.6.0, Python 3.14.4.

The two deltas that drive everything: **20 -> 36 SMs** changes wave
quantization for every tile choice, and **WDDM -> Linux** removes most of the
kernel-launch overhead that the original design was built to hide.

## Setup

```bash
git clone https://github.com/dalberrt/COBRA-GPU-Kernel.git
cd COBRA-GPU-Kernel
git checkout port/blackwell-sm120

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
# cu128 wheels; sm_120 needs PyTorch 2.11+ and Triton 3.6+
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Triton ships with the PyTorch wheel. No compiler toolchain is needed — every
kernel is JIT-compiled by Triton at first use.

## Reproducing the results

```bash
# the full 14-case competition matrix at the competition tolerance
./.venv/bin/python tools/sweep.py --loose

# a subset
./.venv/bin/python tools/sweep.py --loose --cases 1,8,10,14

# correctness checks the shape sweep does not cover
./.venv/bin/python tools/validate.py

# prove the shape-14 oracle is bit-exact with the stock baseline
cd tools && ../.venv/bin/python reference_chunked.py
```

Benchmark hygiene: `sweep.py` runs an 8-second burn-in first, because SM
clocks off idle swing results by up to 4x on this card. Disable with
`--no-burn-in` only when you do not care about the timings.

### The re-derivation tools

These are development tools, not part of the submission. Each one exists
because a specific sm_86 constant needed to be re-measured here:

| tool | question it answers |
|---|---|
| `tools/tune_attn.py` | which attention tile configs suit 36 SMs |
| `tools/confirm_attn.py` | do the sweep's winners survive a head-to-head |
| `tools/tune_gemm.py` | is the Triton GEMM at this card's roofline |
| `tools/diag_gemm.py` | is the thin-K gap the fp32 store or the tiling |
| `tools/diag_small.py` | true in-graph GEMM time, without wrapper overhead |
| `tools/graph_gate.py` | is CUDA-graph capture still worth it on Linux |
| `tools/precision_probe.py` | is the fp32 floor at small d_model still needed |
| `tools/reference_chunked.py` | a memory-safe oracle for shape 14 |

Profiling note: `torch.profiler`'s CUDA tracing does not work on this
driver (`CUPTI_ERROR_INVALID_DEVICE`), and Nsight is blocked by
`RmProfilingAdminOnly: 1` without root. Every number in this repository is
therefore from CUDA events, with small shapes timed **inside a CUDA graph**
because that is how the model executes them.

## Shape 14, which is the interesting one

Test shape 14 is B=32, d=1024, H=16, **S=100000**, L=2. The original reports
it as "not runnable on this hardware", and the diagnosis was right:

* the input tensor is **12.21 GiB** in fp32 and the output is another 12.21
  GiB, against 15.47 GiB of usable VRAM — the pair cannot both be resident,
  and no kernel change alters that;
* the reference `BaselineSelfAttention` materializes `scores[B,H,S,S]` =
  **19,073 GB**, so there is no reference output to compare against *on any
  GPU that exists*.

This port makes it run:

1. **Batch streaming.** `forward()` walks the batch in chunks sized from
   `torch.cuda.mem_get_info()` — the device's *actual* free memory, not a
   baked-in constant — halving the chunk on OOM, and writes into a tensor on
   the input's own device, so a host-resident input streams through the GPU.
   Verified **bit-exact** against the non-streamed path.
2. **A bit-exact oracle.** `tools/reference_chunked.py` rebuilds the
   baseline's arithmetic with the query axis streamed. It is verified
   `torch.equal`-identical to the stock baseline across causal/non-causal,
   padded/unpadded, and non-power-of-two shapes.

Correctness for shape 14 is checked on a subset of the batch at the **full**
S=100000. Every operator in this network is independent across the batch, so
a subset at full sequence length exercises exactly the same code paths — it is
complete coverage of the kernel's behaviour, not a sample of it.

## Limitations, and what I would do with more time

* **No speedup ratio exists for shape 14.** The stock baseline cannot produce
  a reference on any hardware, so shape 14 is reported as correctness plus
  absolute runtime against the chunked oracle, never as a speedup over the
  official baseline. Claiming one would be dishonest.
* **Shape 14 needs a host-resident input.** If the harness allocates x on the
  GPU, 12.21 GiB is gone before this code is reached. That is a property of
  the harness and the 16 GB card, not of the kernel.
* **No Nsight, no CUPTI.** Kernel-level counters (achieved occupancy, register
  counts, memory replays) were unavailable, so occupancy is *modelled* in
  `tools/tune_attn.py` rather than measured. Root access would let the model
  be validated against real counters.
* **The fp32 store makes the large-M GEMMs bandwidth-bound** (99-103% of the
  measured 387 GB/s). An fp16 store would be up to 2.39x faster there, but it
  is the original's accuracy mechanism and the margin against the 2e-3 limit
  is too small to spend without a much larger accuracy campaign.
* **Triton compilation dominates tile search.** Sweeping configs is
  compile-bound, so the search space is pruned analytically first. A persistent
  compile cache across runs would make a much wider search affordable.

## Attribution

Original COBRA design and implementation: the upstream author (see `main` and
`report/TECH_REPORT.md`). Blackwell port, re-derivation, shape-14 streaming
path, and oracle: this branch. See `report/PORT_REPORT.md` for the full
before/after and `report/measurements/` for the raw numbers.
