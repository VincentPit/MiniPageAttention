"""E3 — Prefix sharing.

When N parallel completions share an identical prompt, the block pool needs
only ONE physical copy of the prompt's KV state.

Compares two strategies:

  independent  Each of N sequences runs its own prefill on the (same) prompt.
               Allocates N * ceil(P/bs) prompt blocks per layer.

  shared       One B=1 prefill on a helper cache, then fork_prefix(src_cache=helper)
               into N slots of a B=N decode cache. Allocates ceil(P/bs) prompt
               blocks per layer regardless of N.

Both run a few decode steps to confirm the cache is functional with shared
blocks (output blocks remain per-sequence — only the prompt is shared).

Run from project root:
  python -m cpu.experiments.e3_prefix_sharing
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager, PagedCache  # noqa: E402


def make_aligned_prompt(tokenizer, target_len: int) -> List[int]:
    """Return prompt token ids of EXACTLY target_len tokens."""
    base = "The quick brown fox jumps over the lazy dog. " * 100
    ids = tokenizer(base).input_ids
    if len(ids) < target_len:
        ids = ids * (target_len // len(ids) + 1)
    return ids[:target_len]


def block_bytes(mgr: BlockManager) -> int:
    """Bytes per logical block (K + V across all layers)."""
    elt = mgr.k_pool.element_size()
    return 2 * mgr.n_layers * mgr.block_size * mgr.n_heads * mgr.head_dim * elt


def run_independent(
    model, mgr: BlockManager, prompt_ids: List[int], n_parallel: int, n_decode_steps: int,
) -> Tuple[int, int]:
    """N sequences each run their own prefill on the same prompt.
    Returns (peak_blocks_after_prefill, peak_blocks_after_decode)."""
    device = mgr.k_pool.device
    cache = PagedCache(mgr, batch_size=n_parallel)
    input_ids = torch.tensor([prompt_ids] * n_parallel, dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    past_key_values=cache, use_cache=True)
    peak_pre = mgr.n_used

    next_tokens = out.logits[:, -1, :].argmax(-1, keepdim=True)
    for _ in range(n_decode_steps):
        attn_mask = torch.cat(
            [attn_mask, torch.ones((n_parallel, 1), dtype=torch.long, device=device)], dim=1,
        )
        with torch.no_grad():
            out = model(input_ids=next_tokens, attention_mask=attn_mask,
                        past_key_values=cache, use_cache=True)
        next_tokens = out.logits[:, -1, :].argmax(-1, keepdim=True)
    peak_post = mgr.n_used

    for i in range(n_parallel):
        cache.free_sequence(i)
    return peak_pre, peak_post


def run_shared(
    model, mgr: BlockManager, prompt_ids: List[int], n_parallel: int, n_decode_steps: int,
) -> Tuple[int, int]:
    """One prefill on a helper, fork_prefix into N decode slots."""
    P = len(prompt_ids)
    device = mgr.k_pool.device

    # B=1 helper does the prefill once.
    helper = PagedCache(mgr, batch_size=1)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    past_key_values=helper, use_cache=True)

    # B=N main cache, fork prompt blocks from helper.
    cache = PagedCache(mgr, batch_size=n_parallel)
    for i in range(n_parallel):
        cache.fork_prefix(src_seq=0, dst_seq=i, n_tokens=P, src_cache=helper)
    peak_pre = mgr.n_used

    # First decode token: sample from prompt's last logits with replacement so
    # parallel sequences diverge (otherwise greedy gives identical outputs).
    last_logits = out.logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    next_tokens = torch.multinomial(
        probs, num_samples=n_parallel, replacement=True,
    ).unsqueeze(-1)

    decode_attn = torch.ones((n_parallel, P + 1), dtype=torch.long, device=device)
    for _ in range(n_decode_steps):
        with torch.no_grad():
            out2 = model(input_ids=next_tokens, attention_mask=decode_attn,
                         past_key_values=cache, use_cache=True)
        next_tokens = out2.logits[:, -1, :].argmax(-1, keepdim=True)
        decode_attn = torch.cat(
            [decode_attn, torch.ones((n_parallel, 1), dtype=torch.long, device=device)], dim=1,
        )
    peak_post = mgr.n_used

    for i in range(n_parallel):
        cache.free_sequence(i)
    helper.free_sequence(0)
    return peak_pre, peak_post


def run_e3(prompt_len: int = 256, n_parallel: int = 8, n_decode_steps: int = 4,
           block_size: int = 16) -> bool:
    print("loading gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    cfg = model.config

    prompt_ids = make_aligned_prompt(tokenizer, prompt_len)
    assert len(prompt_ids) == prompt_len

    mgr_kwargs = dict(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=4096,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )

    mgr_indep = BlockManager(**mgr_kwargs)
    indep_pre, indep_post = run_independent(
        model, mgr_indep, prompt_ids, n_parallel, n_decode_steps,
    )

    mgr_shared = BlockManager(**mgr_kwargs)
    shared_pre, shared_post = run_shared(
        model, mgr_shared, prompt_ids, n_parallel, n_decode_steps,
    )

    blocks_per_seq_prompt = math.ceil(prompt_len / block_size)
    n_layers = cfg.n_layer
    expected_indep = n_parallel * blocks_per_seq_prompt * n_layers
    expected_shared = blocks_per_seq_prompt * n_layers
    bytes_per = block_bytes(mgr_indep)

    print(f"\n  P={prompt_len}, N={n_parallel}, decode_steps={n_decode_steps}, "
          f"layers={n_layers}, block_size={block_size}")
    print(f"  bytes/block = {bytes_per/1e6:.2f} MB")
    print(f"  ceil(P/bs) per layer = {blocks_per_seq_prompt}")

    print(f"\n  {'':<24} {'measured':>10} {'expected':>10} {'MB':>8}")
    print(f"  {'indep, after prefill':<24} {indep_pre:>10} {expected_indep:>10} "
          f"{indep_pre*bytes_per/1e6:>8.1f}")
    print(f"  {'shared, after prefill':<24} {shared_pre:>10} {expected_shared:>10} "
          f"{shared_pre*bytes_per/1e6:>8.1f}")
    print(f"  {'indep, after decode':<24} {indep_post:>10} {'':>10} "
          f"{indep_post*bytes_per/1e6:>8.1f}")
    print(f"  {'shared, after decode':<24} {shared_post:>10} {'':>10} "
          f"{shared_post*bytes_per/1e6:>8.1f}")

    print(f"\n  prefill ratio (indep / shared): {indep_pre/shared_pre:.2f}x")
    print(f"  total ratio   (indep / shared): {indep_post/shared_post:.2f}x")

    print(f"\nfinal blocks (should be 0): "
          f"indep={mgr_indep.n_used}, shared={mgr_shared.n_used}")

    correctness = (indep_pre == expected_indep
                   and shared_pre == expected_shared
                   and mgr_indep.n_used == 0
                   and mgr_shared.n_used == 0)
    return correctness


if __name__ == "__main__":
    success = run_e3()
    sys.exit(0 if success else 1)
