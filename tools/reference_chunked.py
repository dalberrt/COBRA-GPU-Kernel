"""A memory-safe reference for shapes the stock baseline cannot run.

Development tool -- not part of the submission.

WHY THIS EXISTS
---------------
`BaselineSelfAttention.forward` materializes the full score tensor:

    scores = torch.matmul(q, k.transpose(-2, -1))      # [B, H, S, S]

At test shape 14 (B=32, H=16, S=100000) that is 32*16*1e10*4 B = 19,073 GB.
A single head's [S, S] block is 37.3 GB. No GPU can run it, so there is no
reference output to diff against and shape 14 is un-gradeable as written.

This module rebuilds the SAME arithmetic with the query dimension streamed in
blocks, so peak memory is O(BLOCK_Q * S) instead of O(B * H * S * S).  It is
deliberately NOT an optimization: it makes no algebraic change, keeps every
cast in the same place, and is slower than the baseline per FLOP.  Its only
job is to be a trustworthy oracle.

EXACTNESS
---------
Matched against `BaselineSelfAttention` op-for-op:
  * scores in x.dtype, scaled by self.scale after the matmul
  * causal mask applied via masked_fill(-inf) BEFORE the softmax
  * key-padding mask applied via masked_fill(-inf) BEFORE the softmax
  * softmax computed in fp32 (`scores.float()`), then cast back to x.dtype
  * context = probs @ v in x.dtype
The only reordering is that the query axis is visited in blocks.  Softmax is
along the key axis, so blocking the query axis is exact -- each query row's
normalization is untouched.  Verified against the true baseline at shapes
where the baseline still fits (see verify_against_baseline below).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "bm", os.path.join(ROOT, "torch_transformer_benchmark.py")
)
bm = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("bm", bm)
if not hasattr(bm, "BaselineTransformer"):
    _spec.loader.exec_module(bm)


def _plan(batch, heads, seq_len, budget_frac=0.25):
    """Pick (head_chunk, block_q) so the score block fits in free VRAM.

    The score block is [batch, head_chunk, block_q, kv_end] and the softmax
    needs roughly three of them live at once (the fp32 upcast, the exp, and
    the result).  kv_end is bounded by seq_len.  Heads are taken one at a time
    so block_q can stay large, which keeps each matmul big enough to be worth
    launching.  At B=2, H=16, S=100000 a fixed block_q=2048 would ask for a
    26 GB score block; this returns 512 instead.
    """
    free, _ = torch.cuda.mem_get_info()
    budget = int(free * budget_frac)
    head_chunk = 1
    per_row = batch * head_chunk * seq_len * 4 * 3
    bq = max(1, budget // max(1, per_row))
    bq = 1 << (bq.bit_length() - 1)          # round down to a power of two
    return head_chunk, max(16, min(2048, bq))


def chunked_attention(
    self,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    block_q: Optional[int] = None,
) -> torch.Tensor:
    """Drop-in for BaselineSelfAttention.forward, streamed over heads and
    query blocks.  Identical arithmetic; only the visitation order differs."""
    batch, seq_len, _ = x.shape
    H = self.num_heads

    q = self._split_heads(self.q_proj(x))          # [B, H, S, hd]
    k = self._split_heads(self.k_proj(x))
    v = self._split_heads(self.v_proj(x))
    context = torch.empty_like(q)

    head_chunk, auto_bq = _plan(batch, H, seq_len)
    bq = block_q or auto_bq

    for h0 in range(0, H, head_chunk):
        h1 = min(h0 + head_chunk, H)
        for q0 in range(0, seq_len, bq):
            q1 = min(q0 + bq, seq_len)
            qb = q[:, h0:h1, q0:q1]

            # Causality bounds the keys this block can attend to at q1, so the
            # score block is [B, hc, Bq, q1] rather than [B, hc, Bq, S].  These
            # are the same finite scores the baseline computes; the entries it
            # would fill with -inf are simply never formed.
            kv_end = q1 if causal else seq_len
            kb = k[:, h0:h1, :kv_end]
            vb = v[:, h0:h1, :kv_end]

            scores = torch.matmul(qb, kb.transpose(-2, -1)) * self.scale

            if causal:
                qpos = torch.arange(q0, q1, device=x.device)[:, None]
                kpos = torch.arange(0, kv_end, device=x.device)[None, :]
                scores = scores.masked_fill(kpos > qpos, float("-inf"))

            if valid_token_mask is not None:
                invalid = ~valid_token_mask[:, None, None, :kv_end]
                scores = scores.masked_fill(invalid, float("-inf"))

            # Same as the baseline: fp32 softmax, cast back to the model dtype.
            probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            context[:, h0:h1, q0:q1] = torch.matmul(probs, vb)
            del scores, probs

    context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
    output = self.out_proj(context)

    if valid_token_mask is not None:
        output = output.masked_fill(~valid_token_mask[..., None], 0)
    return output


def make_chunked_reference(cfg, block_q: Optional[int] = None):
    """A BaselineTransformer whose attention is streamed, not materialized.

    Built from the stock class so `state_dict()` is byte-identical in keys and
    shapes -- `copy_model_weights(chunked, optimized)` works unchanged.
    """
    model = bm.BaselineTransformer(cfg)
    for layer in model.layers:
        attn = layer.attention
        layer.attention.forward = (
            lambda x, m=None, c=False, _a=attn: chunked_attention(_a, x, m, c, block_q)
        )
    return model


def verify_against_baseline(device="cuda", dtype=torch.float32) -> bool:
    """Prove the oracle is an oracle, at shapes the true baseline still fits."""
    shapes = [
        (2, 128, 128, 4, 128, 2),
        (4, 256, 128, 4, 128, 2),
        (2, 512, 256, 8, 256, 2),
        (3, 333, 96, 3, 96, 2),      # S not a multiple of block_q
    ]
    dev = torch.device(device)
    ok = True
    print(f"{'B':>4} {'S':>6} {'D':>5} {'H':>3} {'causal':>7} {'pad':>5} "
          f"{'max_abs':>11} {'bitexact':>9}")
    for B, S, D, H, L in [(s[0], s[1], s[2], s[3], s[5]) for s in shapes]:
        for causal in (True, False):
            for pad in (0.0, 0.3):
                cfg = bm.TransformerConfig(B, S, D, H, D, L, causal)
                torch.manual_seed(1234)
                base = bm.BaselineTransformer(cfg).to(dev, dtype).eval()
                ref = make_chunked_reference(cfg, block_q=128).to(dev, dtype).eval()
                ref.load_state_dict(base.state_dict())
                x, mask = bm.generate_random_case(cfg, dev, dtype, 1234, pad, 1.0)
                with torch.inference_mode():
                    a, b = base(x, mask), ref(x, mask)
                err = (a - b).abs().max().item()
                exact = torch.equal(a, b)
                ok &= err <= 1e-5
                print(f"{B:>4} {S:>6} {D:>5} {H:>3} {str(causal):>7} {pad:>5} "
                      f"{err:>11.3e} {str(exact):>9}")
                del base, ref, x, mask, a, b
                torch.cuda.empty_cache()
    print("\nORACLE VERIFIED" if ok else "\nORACLE MISMATCH")
    return ok


if __name__ == "__main__":
    torch.manual_seed(1234)
    raise SystemExit(0 if verify_against_baseline() else 1)
