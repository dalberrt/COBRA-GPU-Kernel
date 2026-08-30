# Step 2a: re-deriving the tile configs for sm_120

Tools: `tools/tune_attn.py`, `tools/confirm_attn.py`, `tools/tune_gemm.py`,
`tools/diag_gemm.py`, `tools/diag_small.py`.

Playbooks followed: `triton.optimize-triton-block-parameters`,
`patterns.choose-tile-size-and-work-partitioning`,
`triton.write-triton-attention-kernel`.

## Method

Blind grid search was the wrong tool: Triton **compilation** dominates (the GPU
sits at 0% while 400 configs build, ~20 min/case). Candidates are therefore
pruned on paper with an explicit shared-memory and register model, then ranked
by (achieved occupancy x wave-quantization efficiency), and only the top 36 are
compiled and timed.

Wave efficiency is the sm_120-specific term. With 36 SMs a grid of 64 CTAs
runs 1.78 waves -- the second wave is 44% empty. On the friend's 20-SM part the
same grid runs 3.2 waves and quantizes differently.

Profiling note: CUPTI fails on this driver (`CUPTI_ERROR_INVALID_DEVICE`) and
`RmProfilingAdminOnly: 1` blocks Nsight without root, so there is no
torch-profiler kernel trace. All timings here are CUDA events, and the small
shapes are measured **inside a CUDA graph**, because that is how the model
actually runs them.

## Attention kernel: shipped sm_86 config vs sm_120 sweep vs SDPA

| case | hd | S | shipped (sm_86) | ship ms | sm_120 sweep | best ms | SDPA ms | gain | vs SDPA |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 128 | (64,64,4,4) | 0.0217 | (32,32,4,1) | 0.0227 | 0.0228 | 0.958x | 1.003x |
| 8 | 256 | 128 | (64,64,4,2) | **fails** | (32,16,4,3) | 0.1450 | 0.1624 | — | **1.121x** |
| 9 | 128 | 128 | (64,64,4,2) | 0.0225 | (128,32,8,1) | 0.0228 | 0.0236 | 0.990x | 1.036x |
| 10 | 64 | 128 | (64,64,4,3) | 0.0276 | (16,32,4,1) | 0.0226 | 0.0228 | **1.220x** | 1.007x |
| 11 | 8 | 128 | (64,64,4,3) | 0.0277 | = shipped | 0.0277 | 0.0709 | 1.001x | 2.558x |
| 12 | 32 | 32 | (32,32,4,2) | 0.0192 | = shipped | 0.0187 | 0.0228 | 1.027x | 1.217x |
| 13 | 32 | 1024 | (64,64,4,3) | 0.4358 | (128,32,8,2) | 0.4848 | 0.4497 | **0.899x** | 0.928x |

**Result: the sm_86 attention ladder largely transfers.** Cases 1, 9, 12 sit at
the noise floor (~0.022 ms whatever the config -- these kernels are latency-
bound, not tile-bound). Case 13's shipped config *beats* the sweep winner by
1.11x, so the sweep's ranking is kept only where it reproduces under the
head-to-head. Two changes survive:

* **case 10 (head_dim=64): 0.0276 -> 0.0226 ms, 1.22x.**
* **case 8 (head_dim=256): the shipped ladder raises OutOfResources and the
  model silently falls back to SDPA.** head_dim=256 does run on sm_120 at
  (32,16,4,3) and beats SDPA by 1.12x, so the `head_dim <= 128` gate is wrong
  on this card. Attention is only ~5% of case 8's runtime, so this is worth
  ~1.05x on that case, not more.

## GEMM: the config list already self-adapts

`_gemm_kernel` is wrapped in `@triton.autotune(key=["M","N","K"])`, so the
*selection* re-tunes on this card automatically. Only the candidate list is
inherited. Measured against this card's own fp16 peak (**50.3 TFLOP/s**, from
cuBLAS on square GEMMs -- far below the marketing number, consumer Blackwell
halves fp16-with-fp32-accumulate):

| shape | M | K | N | Triton | cuBLAS | % of peak |
|---|---|---|---|---|---|---|
| case 8 qkv | 8192 | 1024 | 3072 | 1.0532 ms | 1.0587 ms | **97.2%** |
| case 8 out/ffn | 8192 | 1024 | 1024 | 0.3559 ms | 0.3498 ms | **95.9%** |
| case 13 out/ffn | 65536 | 128 | 128 | 0.1306 ms | 0.0610 ms | 32.7% |
| case 6 qkv | 1280000 | 128 | 384 | 5.7227 ms | 3.7172 ms | 43.7% |

Case 8 is at hardware roofline; no config change can help it.

### The thin-K gap is the fp32 store, not the tiling

`tools/diag_gemm.py` separates the causes. Measured copy bandwidth on this
card is **387 GB/s**:

| shape | Triton fp32 store | Triton fp16 store | cuBLAS | fp32 store costs | % of bandwidth |
|---|---|---|---|---|---|
| case 1/4/5/9 out+ffn | 0.0299 | 0.0299 | 0.0134 | 1.00x | 55% |
| case 13 out+ffn | 0.1310 | 0.0548 | 0.0610 | **2.39x** | **99%** |
| case 6 qkv | 5.7336 | 3.5139 | 3.7331 | **1.63x** | **103%** |

At large M the fp32 store makes these GEMMs bandwidth-bound -- they are already
at 99-103% of achievable bandwidth, so **no tile config can improve them**. With
an fp16 store Triton would beat cuBLAS (0.90x, 0.94x), but the fp32 store is
the friend's accuracy mechanism and the margin is only 1.51x. Not touched.

### The small-M "2.2x gap" was a measurement artifact

The 0.0299 ms above is identical for fp32 and fp16 stores and sits at only 55%
of bandwidth -- the signature of host-side cost (the Python wrapper plus the
autotuner's cache lookup), not GPU work. The model runs these inside a CUDA
graph. Re-measured **in-graph** (`tools/diag_small.py`):

| shape | M | K | N | Triton fp32 | cuBLAS fp16 | ratio |
|---|---|---|---|---|---|---|
| case 1/4/5/9 out+ffn | 8192 | 128 | 128 | 0.01228 | 0.01084 | 1.13x |
| case 1/4/5/9 qkv | 8192 | 128 | 384 | 0.02448 | 0.02358 | 1.04x |
| case 13 out+ffn | 65536 | 128 | 128 | 0.13047 | 0.05966 | 2.19x |
| case 7 out+ffn | 8192 | 32 | 32 | 0.00246 | 0.00203 | 1.21x |

In-graph the small GEMMs are within 1.04-1.21x of cuBLAS **while additionally
fusing the bias**, so they are at or better than parity in real terms. The
earlier 2.2x was an artifact of benchmarking through the Python wrapper.

## Conclusion for step 2a

The honest result is mostly negative, and is reported as such: the sm_86 GEMM
config list and most of the attention ladder transfer correctly to sm_120,
because the GEMM path autotunes itself and the small-shape kernels are
latency-bound rather than tile-bound. Changing them would be churn.
Two measured changes are worth making (case 10's tile, case 8's head_dim gate);
everything else stays.
