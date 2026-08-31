# Baseline: friend's sm_86-tuned kernel, measured on sm_120

Branch point: `308e46f` (main, unmodified).
Command: `python tools/sweep.py --loose`
Gate: abs<=0.002 OR rel<=0.02 | dtype=float32 | padding=0.0 | precision=auto

## Hardware (measured, not assumed)

| Property | Friend's sm_86 (RTX 3050, Windows/WDDM) | This machine: sm_120 (RTX 5060 Ti, Linux) |
|---|---|---|
| SMs | 20 | **36** (1.8x more) |
| Registers / SM | 65536 | 65536 |
| Shared mem opt-in / block | 101376 B | 101376 B |
| Shared mem / SM | 102400 B | 102400 B |
| Max threads / SM | 1536 | 1536 |
| L2 cache | ~2 MB | **32 MB** (16x more) |
| VRAM | 8 GB | 15.47 GiB usable |
| Driver model | **WDDM** (high launch overhead) | **Linux** (low launch overhead) |

The friend's target is recorded in the source itself
(torch_transformer_benchmark.py:176, "RTX 3050, sm_86, Windows/WDDM"), not
an RTX 3090. Two consequences drive this whole port:

1. **Launch overhead.** His design note says the baseline is "CPU-DISPATCH-bound,
   not GPU-bound" and that collapsing ~115 dispatches into one CUDA graph is
   "the first-order win". That is a WDDM property. On Linux a kernel launch is
   single-digit microseconds, so the graph gate has to be re-derived rather
   than inherited (see 2c).
2. **Occupancy.** 20 SMs -> 36 SMs changes wave quantization for every tile
   choice. Grids that filled his card leave mine partly idle, and vice versa.
   His register-pressure ladder was tuned against a 20-SM machine.

Torch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8, Python 3.14.4, driver 595.84.

## Result: 13/13 pass, median 7.87x, case 14 ERROR

| # | B | D | H | S | dt | graph | res | max_abs | base ms | opt ms | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 128 | 4 | 128 | fp16 | Y | PASS | 1.017e-03 | 1.984 | 0.394 | 5.04x |
| 2 | 1 | 128 | 4 | 128 | fp16 | Y | PASS | 7.181e-04 | 0.970 | 0.061 | 15.91x |
| 3 | 4 | 128 | 4 | 128 | fp16 | Y | PASS | 8.096e-04 | 0.936 | 0.081 | 11.48x |
| 4 | 16 | 128 | 4 | 128 | fp16 | Y | PASS | 1.047e-03 | 0.942 | 0.120 | 7.87x |
| 5 | 128 | 128 | 4 | 128 | fp16 | Y | PASS | 1.076e-03 | 7.121 | 0.761 | 9.35x |
| 6 | 10000 | 128 | 4 | 128 | fp16 | n | PASS | 1.326e-03 | 729.967 | 79.428 | 9.19x |
| 7 | 64 | 32 | 4 | 128 | fp32 | Y | PASS | 1.135e-03 | 1.432 | 0.195 | 7.35x |
| 8 | 64 | 1024 | 4 | 128 | fp16 | Y | PASS | 1.072e-03 | 29.795 | 10.354 | **2.88x** |
| 9 | 64 | 128 | 1 | 128 | fp16 | Y | PASS | 1.230e-03 | 1.226 | 0.419 | **2.92x** |
| 10 | 64 | 128 | 2 | 128 | fp16 | Y | PASS | 1.045e-03 | 1.495 | 0.378 | **3.95x** |
| 11 | 64 | 128 | 16 | 128 | fp16 | Y | PASS | 1.162e-03 | 10.946 | 0.438 | 25.01x |
| 12 | 64 | 128 | 4 | 32 | fp16 | Y | PASS | 1.025e-03 | 0.949 | 0.132 | 7.19x |
| 13 | 64 | 128 | 4 | 1024 | fp16 | Y | PASS | 1.321e-03 | 172.990 | 5.173 | 33.44x |
| 14 | 32 | 1024 | 16 | 100000 | — | — | **ERROR** | OutOfMemoryError: tried to allocate 12.21 GiB | | | |

Worst max_abs across all passing cases: **1.326e-03** vs the 2e-3 atol limit (margin 1.51x).

## Weak cases on THIS card (tuning targets)

Cases 8 (2.88x), 9 (2.92x), 10 (3.95x) are the low outliers. All three are
low-parallelism shapes: case 9/10 have H=1/H=2, so the attention grid is
(ceil(S/BM), H, B) = (1,1,64) / (1,2,64) -- 64 and 128 CTAs against 36 SMs.
Case 8 is d_model=1024 with only 4 heads. These are where the sm_86 config
ladder transfers worst.

## Case 14 root cause (corrects the brief's assumption)

The OOM is NOT in the friend's kernel. It is in `generate_random_case`:
the input tensor x[32, 100000, 1024] in fp32 is 12.21 GiB by itself, and
the harness allocates it before the model is ever called.

Further, the *reference* `BaselineSelfAttention` materializes
`scores = [B, H, S, S]` explicitly (torch_transformer_benchmark.py:96):

    32 * 16 * 100000 * 100000 * 4 B = 19,073 GB

A single head's [S,S] score matrix is 37.3 GB. So the baseline cannot run
shape 14 on any GPU that exists; there is no reference output to diff
against. See 01_case14_decision.md.
