"""Paged-attention decode kernel — Phase 3.

Mirrors vLLM's paged_attention_v1 / v2 in spirit: takes a paged KV pool,
per-sequence block_tables, and seq_lens, and computes attention output
*directly from paged memory* — no _gather() into a contiguous [B, H, L, D].

Two backends share a signature:
  - `paged_attention_decode_torch` — torch-native, no kernel; the local
    sanity-check path (this Mac has no CUDA).
  - `paged_attention_decode_triton` — Triton kernel for CUDA. v1-style
    single-pass online softmax (one program per (batch, head)). v2-style
    partition+reduce is a future optimization for very long contexts.

Signature mirrors the vLLM v2 launcher's relevant arguments. We omit
FP8 KV cache, ALiBi, block-sparse, GQA scales, and tp_rank — none apply
to GPT-2 fp16 decode and they would obscure the core access pattern.

Layout note: vLLM splits key_cache as [num_blocks, num_heads, head/x,
block, x] for memory coalescing. We use the simpler
[num_blocks, block_size, num_heads, head_dim] that BlockManager already
produces. The Triton kernel handles strides explicitly so the layout
choice is local to this file.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Torch reference: same access pattern as the kernel, no gather.
# ---------------------------------------------------------------------------

def paged_attention_decode_torch(
    query: torch.Tensor,        # [B, H, D]
    key_cache: torch.Tensor,    # [n_blocks, block_size, H, D]
    value_cache: torch.Tensor,  # [n_blocks, block_size, H, D]
    block_tables: torch.Tensor, # [B, max_blocks_per_seq] int32, -1 padded
    seq_lens: torch.Tensor,     # [B] int32
    scale: float,
) -> torch.Tensor:
    B, H, D = query.shape
    out = torch.empty_like(query)
    bs = key_cache.shape[1]

    seq_lens_list = seq_lens.tolist()
    for b in range(B):
        L = int(seq_lens_list[b])
        if L == 0:
            out[b].zero_()
            continue
        n_blocks = (L + bs - 1) // bs
        bids = block_tables[b, :n_blocks].to(torch.long)
        # [n_blocks, bs, H, D] -> [n_blocks*bs, H, D] -> trim to L
        K = key_cache[bids].reshape(-1, H, D)[:L]    # [L, H, D]
        V = value_cache[bids].reshape(-1, H, D)[:L]  # [L, H, D]
        # Compute per-head: q [H, D] @ K^T [H, D, L] -> [H, L]
        q = query[b]                              # [H, D]
        logits = torch.einsum("hd,lhd->hl", q, K) * scale   # [H, L]
        probs = torch.softmax(logits, dim=-1)               # [H, L]
        out[b] = torch.einsum("hl,lhd->hd", probs, V)       # [H, D]
    return out


# ---------------------------------------------------------------------------
# Triton kernel (CUDA only).
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    # Tuning space: (num_warps, num_stages). BLOCK_SIZE is fixed by the cache
    # layout, D is fixed by the model — both are constexprs and serve as the
    # autotune key, so we tune once per (head_dim, page_size) and cache.
    # Config dict is empty because we don't expose tunable kwargs; the
    # decorator's job here is just to sweep launch parameters.
    _AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (1, 2, 3)
    ]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["D", "BLOCK_SIZE"])
    @triton.jit
    def _paged_attn_decode_kernel(
        out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
        block_tables_ptr, seq_lens_ptr,
        scale,
        # strides
        q_stride_b, q_stride_h,
        kv_stride_block, kv_stride_slot, kv_stride_h,
        out_stride_b, out_stride_h,
        bt_stride_b,
        # runtime
        max_blocks,
        # constants
        D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        seq_len = tl.load(seq_lens_ptr + b)

        d_off = tl.arange(0, D)
        q_load = tl.load(q_ptr + b * q_stride_b + h * q_stride_h + d_off)
        q = q_load.to(tl.float32) * scale

        m_i = tl.full((), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((), dtype=tl.float32)
        acc = tl.zeros((D,), dtype=tl.float32)

        slot_off = tl.arange(0, BLOCK_SIZE)

        # Iterate over all possible blocks. Out-of-range blocks (slot_pos >=
        # seq_len) get fully masked, contributing 0 to the online softmax —
        # see the math in the file docstring for why this is safe even when
        # m_i is -inf on the first iteration (first block is always in-range
        # because the caller asserts seq_len > 0).
        for blk_idx in range(0, max_blocks):
            slot_pos = blk_idx * BLOCK_SIZE + slot_off          # [BLOCK_SIZE]
            slot_mask = slot_pos < seq_len                       # [BLOCK_SIZE]

            bid = tl.load(block_tables_ptr + b * bt_stride_b + blk_idx)
            # Padded entries in block_tables are -1; keep the address valid
            # by clamping. Their slot_mask is all-False so the load is masked.
            bid_safe = tl.maximum(bid, 0)

            kv_base = (
                bid_safe * kv_stride_block
                + slot_off[:, None] * kv_stride_slot
                + h * kv_stride_h
                + d_off[None, :]
            )
            k_tile = tl.load(
                k_cache_ptr + kv_base, mask=slot_mask[:, None], other=0.0,
            ).to(tl.float32)
            v_tile = tl.load(
                v_cache_ptr + kv_base, mask=slot_mask[:, None], other=0.0,
            ).to(tl.float32)

            s = tl.sum(k_tile * q[None, :], axis=1)              # [BLOCK_SIZE]
            s = tl.where(slot_mask, s, -float("inf"))

            m_blk = tl.max(s, axis=0)
            m_new = tl.maximum(m_i, m_blk)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            p = tl.where(slot_mask, p, 0.0)

            acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(
            out_ptr + b * out_stride_b + h * out_stride_h + d_off,
            out.to(q_load.dtype),
        )


def paged_attention_decode_triton(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not _HAS_TRITON:
        raise RuntimeError("Triton not installed; use paged_attention_decode_torch")
    if not query.is_cuda:
        raise RuntimeError("Triton kernel requires CUDA tensors")

    B, H, D = query.shape
    bs = key_cache.shape[1]
    max_blocks = block_tables.shape[1]
    out = torch.empty_like(query)

    grid = (B, H)
    _paged_attn_decode_kernel[grid](
        out, query, key_cache, value_cache,
        block_tables, seq_lens,
        scale,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        max_blocks,
        D=D, BLOCK_SIZE=bs,
    )
    return out


# ---------------------------------------------------------------------------
# Dispatcher: pick backend by device + availability.
# ---------------------------------------------------------------------------

def paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Compute O[b,h,:] = softmax(scale * q[b,h,:] @ K_b^T) @ V_b
    where (K_b, V_b) live in paged memory addressed by block_tables[b]
    and have length seq_lens[b].

    backend: "torch" | "triton" | None (auto).
    """
    if backend is None:
        backend = "triton" if (_HAS_TRITON and query.is_cuda) else "torch"
    if backend == "triton":
        return paged_attention_decode_triton(
            query, key_cache, value_cache, block_tables, seq_lens, scale,
        )
    if backend == "torch":
        return paged_attention_decode_torch(
            query, key_cache, value_cache, block_tables, seq_lens, scale,
        )
    raise ValueError(f"unknown backend {backend!r}")


def naive_attention_for_test(
    query: torch.Tensor,        # [B, H, D]
    key_full: torch.Tensor,     # [B, L, H, D]  (variable L per batch -> pad with zeros)
    value_full: torch.Tensor,   # same
    seq_lens: torch.Tensor,     # [B]
    scale: float,
) -> torch.Tensor:
    """Reference for the test harness: contiguous K/V, mask pad."""
    B, H, D = query.shape
    L = key_full.shape[1]
    pos = torch.arange(L, device=query.device)[None, :]   # [1, L]
    mask = pos < seq_lens.to(query.device)[:, None]       # [B, L]
    # logits [B, H, L]
    logits = torch.einsum("bhd,blhd->bhl", query, key_full) * scale
    logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("bhl,blhd->bhd", probs, value_full)
    return out
