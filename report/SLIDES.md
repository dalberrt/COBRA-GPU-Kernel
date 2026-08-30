# Slide deck content — COBRA GPU Kernel

Everything below is measured on the machine described in slide 2. Numbers in **bold** are the ones
worth putting on the slide itself; the rest is speaker material.

Suggested length: 14 slides / ~12 minutes. Slides 4, 5, 8 and 11 are the ones that carry the talk —
if time is cut, cut 3, 7, 13.

---

## 1 — Title

**Optimizing a Transformer forward pass on a consumer GPU**
11.78x median speedup, 13/13 cases correct, zero failing elements

Subtitle: RTX 3050 8 GB · PyTorch 2.9 · Triton · 5 custom kernels

---

## 2 — The task and the hardware

| | |
|---|---|
| Target | Pre-LN Transformer forward, `UserOptimizedTransformer` |
| Correctness gate | `abs_err <= 0.002` **OR** `rel_err <= 0.02`, **zero** failing elements |
| Test matrix | 14 shapes; `ffn_dim == d_model`, `causal=True`, 4 layers |
| GPU | RTX 3050 8 GB, sm_86, 20 SMs, 224 GB/s, 2 MB L2 |
| Measured peak | **17.56 TFLOPS** fp16 tensor · 9.00 TFLOPS TF32 |

Speaker note: the gate is stricter than `torch.allclose`, which would allow `atol + rtol*|ref|`.
One bad element out of ~5 million fails the case.

---

## 3 — The environment was the first blocker

The provided benchmark **could not run at all** on the machine's PyTorch 1.13:

```
compare_outputs()  ->  failed_mask.any(dim=reduce_dims)
TypeError: any() received an invalid combination of arguments - got (dim=tuple,)
```

Multi-dimensional `Tensor.any` landed in PyTorch 2.0. Every invocation died before printing a result.

Fixed by building an isolated venv (torch 2.9.1+cu128) — the global environment pins
pytorch-lightning / transformers against 1.13 and could not be upgraded in place.

Speaker note: found by *running* the harness rather than reading it. Worth one line: the upgrade was
a prerequisite, not an optimization.

---

## 4 — The measurement that decided the whole design ★

We timed the CPU cost of *queuing* a forward against the total time including the synchronize:

| Case | CPU dispatch / call | Total / call | Verdict |
|---|---|---|---|
| B=1 | **4.135 ms** | **4.136 ms** | 100 % CPU-bound |
| B=64, S=32 | 3.846 ms | 3.846 ms | 100 % CPU-bound |
| B=64 | 9.454 ms | 10.821 ms | 87 % CPU-bound |

**The GPU was idle ~99.8 % of the forward.** There is a ~3.3 ms floor independent of batch size —
cases with a 16x spread in work all land at 3.3–3.8 ms, because the baseline issues ~115 GPU ops
per call.

> The bottleneck was the **number of PyTorch dispatches**, not tensor-core utilization.

Speaker note: this is the slide to spend time on. The intuitive move — "optimize a Transformer" →
fused kernels and tensor cores — would have been the wrong place to start. Suggested visual: two
stacked bars, GPU-busy vs GPU-idle, for B=1.

---

## 5 — Approach: five Triton kernels + one graph ★

Suggested visual: the layer diagram, baseline (~29 kernels/forward) above, ours (17) below.

| Kernel | Does |
|---|---|
| `_qkvattn_kernel` | projects Q,K,V **inside** the attention CTA + causal FlashAttention-2 |
| `_attn_kernel` | causal attention reading a packed QKV buffer (shapes the above can't serve) |
| `_gemm_kernel` | fp16 in / **fp32 accumulate / fp32 store** + bias + erf-GELU + residual + mask |
| `_gemm_ln_kernel` | the same, **plus the following LayerNorm** in the same epilogue |
| `_add_ln_kernel` | residual-add + mask + LayerNorm + cast |

All of it captured in **one CUDA graph** and replayed.

---

## 6 — Win 1: CUDA graphs

~115 dispatches → **1 graph replay**. Warm-up on a side stream so cuBLAS/Triton workspaces and all
autotuning finish *outside* the capture; any capture failure degrades silently to eager.

**Case 2 (B=1): 4.52 ms → 0.090 ms = 50x.**

Speaker note: this is why the small cases show such large ratios, and it is honest to say so — see
slide 13.

---

## 7 — Win 2: the fp32-store GEMM (the precision unlock) ★

fp16 GEMMs run 2x TF32 on this card, but naive fp16 **fails the gate**. The reason is not the
inputs — fp16 and TF32 both carry 10 explicit mantissa bits — it is that cuBLAS also **writes an
fp16 result**, and PyTorch has no way to ask it for a wider output.

Isolating just the store, same fp16 inputs:

| GEMM | output error |
|---|---|
| cuBLAS fp16 operands, fp16 store | **2.8e-3** |
| Triton fp16 operands, fp32 accumulate, **fp32 store** | **~1e-6** |

The whole error budget is 2e-3, so the cuBLAS store alone consumed it. A custom kernel fixes this
because the accumulator is already fp32 — only the store narrows.

**Effect: fp16 became usable on 12 of 13 cases. Median 2.96x → 5.19x.**

---

## 8 — Win 3: there is no FlashAttention in this build ★

```
torch.__config__.show()       -> USE_FLASH_ATTENTION absent
can_use_flash_attention(...)  -> False for head_dim in {8,32,64,128,256}
```

Every `scaled_dot_product_attention` call was silently running the CUTLASS **memory-efficient**
kernel — instantiated with `kMaxK=64` (wasting ~8x of its accumulator at head_dim=8) and with
**no tensor-core path at all in fp32**.

Replaced with a Triton causal FlashAttention-2 kernel: fp32 online softmax, reads Q/K/V straight
from the packed buffer with computed strides, `head_dim` padded only to 16.

**Case 7 +69 % · case 11 +41 % · case 13 +27 %.**

Speaker note: a good "measure, don't assume" beat — the report had claimed FlashAttention for two
revisions before this was checked.

---

## 9 — Win 4: the fusion cascade

Each step removes kernels *and* DRAM round-trips of the `[tokens, d_model]` residual:

| step | kernels/layer |
|---|---|
| residual-add + mask + LayerNorm + cast → one kernel | 11 → 8 |
| residual-add + mask folded into the GEMM epilogue | 8 → 6 |
| **LayerNorm folded into the GEMM epilogue** | **6 → 4** |
| QKV folded into the attention kernel | removes the `[tokens, 3D]` intermediate |

The last one works because every Pre-LN LayerNorm is preceded by a residual add, so the loop
normalizes with the *next* block's weights.

**~20 % off optimized time on nearly every case.**

---

## 10 — Results ★

| # | shape | speedup | | # | shape | speedup |
|---|---|---|---|---|---|---|
| 1 | B=64 | 8.20x | | 8 | d=1024 | **2.67x** |
| 2 | B=1 | **50.12x** | | 9 | H=1 | 3.94x |
| 3 | B=4 | 34.30x | | 10 | H=2 | 5.89x |
| 4 | B=16 | 12.76x | | 11 | H=16 | 17.39x |
| 5 | B=128 | 7.98x | | 12 | S=32 | 11.78x |
| 6 | B=10000 | 12.90x | | 13 | S=1024 | 29.81x |
| 7 | d=32 | 7.91x | | 14 | S=100000 | not runnable |

**Median 11.78x · geometric mean 11.41x · 13/13 pass, zero failing elements**

Speaker note: quote the **geometric mean** — it is the right average for ratios. The arithmetic mean
(15.82x) is inflated by case 2.

---

## 11 — Correctness ★

| campaign | trials | worst error | margin | failures |
|---|---|---|---|---|
| padding 0.0 | 60 / case | 1.466e-3 | 1.36x | **0** |
| padding 0.3 | 40 / case | 1.466e-3 | 1.36x | **0** |
| ~1000 trials total | | | | **0** |

Plus: `state_dict` byte-identical (so the harness's `strict=True` load still works), padding 0.1–0.99,
all three dtypes, the harness's own default shape, and degenerate shapes (S=1, non-power-of-two).

**Two correctness arguments we proved rather than assumed:**
1. Under causal attention with left-aligned padding, the key-padding mask is a **no-op for every
   valid query** — so no attention mask is ever built.
2. When the harness runs fp16/bf16, the reference's own rounding *is* the target; being **more
   accurate than it scores as error**. Those runs use exact baseline arithmetic and keep only the
   graph win → bit-identical output.

Speaker note: the tail grows with trial count (case 1: 1.100e-3 at 8 trials → 1.314e-3 at 60), so
low-trial-count accuracy claims are not evidence. That finding killed one optimization.

---

## 12 — What we rejected

Credibility slide. Each of these was measured and dropped:

| rejected | why |
|---|---|
| Persistent megakernel | its gate excluded case 4 — which *is* the median |
| out_proj → attention epilogue | `UnboundLocalError` on 7 cases |
| Row-resident block fusion | would have dropped the median to **8.55x** while claiming a gain |
| `torch.compile` | picks kernels by wall-clock tie-break — cannot hold a zero-failure gate |
| bfloat16 | 8 mantissa bits; fails by ~30 000 elements |
| cuBLASLt fused GELU | it is the tanh approximation, ~2.7e-4/layer systematic |
| Longer attention tile | predicted +0–5 %, **measured −3 %** |

---

## 13 — Honest limits

- **Windows WDDM amplifies these ratios.** Case 2's 50x is mostly dispatch overhead removal. On
  Linux the small cases would shrink; the attention, memory and fp16 wins would hold.
- **Case 8 at 2.67x is a real floor** — 28.0 ms against a 23.4 ms arithmetic floor.
- **Case 6 is memory-state dependent**: 12.90x at the harness default, 33.4x with one accuracy
  trial, because the baseline peaks at 11.8 GB on an 8 GB card and fragments VRAM.
- **Case 14 (S=100000) is not runnable** — 12.2 GB input, allocated before our code is reached. No
  ratio is claimed.

Speaker note: putting this slide in *raises* credibility. Judges discount decks that only show wins.

---

## 14 — AI-assisted workflow (bonus)

Built with Claude Code driving profiling, implementation and review.

- **Measure before optimizing.** Profiling overturned the obvious plan (slide 4).
- **Multi-agent adversarial review.** Proposals were generated in parallel, then each was attacked
  by three independent reviewers (numerics / graph-safety / is-the-speedup-real).
- **It earned its keep by rejecting things**, not generating them: it caught a patch that would have
  crashed 11 of 13 cases, one that would have silently dropped the median to 8.55x, and a claimed
  gain that was inflated ~9x.
- **Measurement kept overruling the reviewers too** — they predicted +0–5 % for one change that
  measured −3 %, and discounted another to "flat" that measured +26 %.

Closing line: the loop was *characterize → hypothesize → measure → keep or discard*, and every number
in the deck came off this machine.

---

## Appendix — one-liners for Q&A

- **"Why is case 8 only 2.67x?"** It is the one genuinely compute-bound case; the baseline already
  ran at ~26 % of fp16 peak, and we are at 81 % of the arithmetic roof.
- **"Is this just mixed precision?"** No — precision alone fails the gate. The fp32 *store* is what
  makes fp16 legal, and the largest single win is dispatch removal.
- **"Would this transfer to an A100?"** The kernel work yes; the dispatch win shrinks. Case 14 would
  become runnable.
- **"How do you know it is correct?"** ~1000 trials, zero failing elements, plus `state_dict`
  compatibility and degenerate-shape coverage, all reproducible via `tools/`.
