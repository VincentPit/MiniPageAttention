"""E4 — Throughput.

Two parts:

  1. Batched correctness
     Different-length prompts in a single B=N batch must produce identical
     output tokens to running each prompt as B=1. Validates the scheduler's
     left-padding + attention_mask path; without this, throughput numbers
     measure a model that's computing the wrong thing.

  2. Throughput sweep
     Tokens/sec across batch sizes for the paged scheduler. With v1's
     uniform-batch limitation, every sequence in a batch advances together,
     so this measures paged scheduling under offline-batch conditions.
     Continuous batching (Phase 2) is what unlocks higher steady-state
     throughput on mixed-length workloads.

Run from project root:
  python -m cpu.experiments.e4_throughput
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from cpu.scheduler import Scheduler  # noqa: E402
from bench.workloads import Request  # noqa: E402


BLOCK_SIZE = 16


def make_manager(cfg, n_blocks: int) -> BlockManager:
    return BlockManager(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=n_blocks,
        block_size=BLOCK_SIZE,
        dtype=torch.float32,
        device="cpu",
    )


def part_correctness(model, tokenizer) -> bool:
    """B=N batch with mixed-length prompts vs B=1 references."""
    print("=== Part 1: Batched correctness ===")
    prompts = [
        "Hello",
        "Once upon a time",
        "The quick brown fox jumps over",
        "In the beginning, the universe",
    ]
    max_new = 8

    # B=1 references
    mgr_ref = make_manager(model.config, n_blocks=64)
    sched_ref = Scheduler(model, tokenizer, mgr_ref, max_output_len=max_new)
    references: List[List[int]] = []
    for p in prompts:
        ids = tokenizer(p).input_ids
        req = Request(
            request_id="ref",
            prompt_len=len(ids),
            output_len=max_new,
            prompt_tokens=ids,
        )
        references.append(sched_ref.run_batch([req])[0].output_tokens)

    # B=N batch (left-padded internally by the scheduler)
    mgr_batch = make_manager(model.config, n_blocks=128)
    sched_batch = Scheduler(model, tokenizer, mgr_batch, max_output_len=max_new)
    reqs = []
    for i, p in enumerate(prompts):
        ids = tokenizer(p).input_ids
        reqs.append(Request(
            request_id=f"batch-{i}",
            prompt_len=len(ids),
            output_len=max_new,
            prompt_tokens=ids,
        ))
    batched = sched_batch.run_batch(reqs)

    n_pass = 0
    for i, (ref_toks, b) in enumerate(zip(references, batched)):
        # Truncate to common length so EOS-shortening doesn't cause spurious mismatch.
        L = min(len(ref_toks), len(b.output_tokens))
        match = ref_toks[:L] == b.output_tokens[:L]
        n_pass += int(match)
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {prompts[i]!r}")
        if not match:
            print(f"          ref:     {ref_toks}")
            print(f"          batched: {b.output_tokens}")
            for j in range(L):
                if ref_toks[j] != b.output_tokens[j]:
                    print(f"          diverge at j={j}: ref={ref_toks[j]} batched={b.output_tokens[j]}")
                    break

    print(f"  {n_pass}/{len(prompts)} prompts matched")
    return n_pass == len(prompts) and mgr_batch.n_used == 0


def part_throughput(model, tokenizer) -> None:
    """Tokens/sec sweep across batch sizes."""
    print("\n=== Part 2: Throughput sweep ===")
    prompt_len = 32
    output_len = 16
    batch_sizes = [1, 2, 4, 8]

    # Pool sized for the largest batch. BlockManager allocates independently
    # per layer from one shared free list, so n_blocks must scale with n_layers.
    max_blocks_per_seq = math.ceil((prompt_len + output_len) / BLOCK_SIZE)
    n_blocks = max(batch_sizes) * max_blocks_per_seq * model.config.n_layer + 32

    mgr = make_manager(model.config, n_blocks=n_blocks)
    sched = Scheduler(model, tokenizer, mgr, max_output_len=output_len, seed=0)

    # Warmup: first call carries JIT / cache-warming overhead.
    warmup_req = Request(
        request_id="warmup",
        prompt_len=prompt_len,
        output_len=4,
        prompt_tokens=None,
    )
    sched.run_batch([warmup_req])

    print(f"  prompt_len={prompt_len}, output_len={output_len}")
    print(f"  {'B':>3} {'wall(s)':>9} {'out tokens':>11} {'tok/s total':>13} {'tok/s/seq':>11}")
    for B in batch_sizes:
        # Fresh requests; synthetic random tokens get filled in by the scheduler.
        reqs = [
            Request(
                request_id=f"{B}-{i}",
                prompt_len=prompt_len,
                output_len=output_len,
                prompt_tokens=None,
            )
            for i in range(B)
        ]
        t0 = time.time()
        completions = sched.run_batch(reqs)
        elapsed = time.time() - t0
        total_out = sum(c.output_len for c in completions)
        per_seq = total_out / B / elapsed
        per_total = total_out / elapsed
        print(f"  {B:>3} {elapsed:>9.3f} {total_out:>11} "
              f"{per_total:>13.2f} {per_seq:>11.2f}")

    print(f"\n  blocks remaining after sweep: {mgr.n_used}  (expect 0)")


if __name__ == "__main__":
    print("loading gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    ok = part_correctness(model, tokenizer)
    part_throughput(model, tokenizer)
    sys.exit(0 if ok else 1)
