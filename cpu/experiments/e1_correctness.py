"""E1 — Correctness: PagedCache vs HF DynamicCache.

For each test prompt, runs greedy generation through two paths:
  - reference: model.generate() with default DynamicCache
  - paged:    Scheduler.run_batch with PagedCache, B=1

Pass criterion: identical generated token sequences.

Run with B=1 to isolate cache-correctness from any padding-related issues.
A separate batched test (later) will exercise the padding path.

Run from project root:
  python -m cpu.experiments.e1_correctness
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Make project root importable when run as script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from cpu.scheduler import Scheduler  # noqa: E402
from bench.workloads import Request  # noqa: E402


def reference_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    """HF default greedy generate. Returns list of new (post-prompt) token ids."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0, ids.shape[1]:].tolist()


def paged_generate(scheduler, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    """Greedy generate via Scheduler + PagedCache (B=1)."""
    prompt_ids = tokenizer(prompt).input_ids
    req = Request(
        request_id="e1",
        prompt_len=len(prompt_ids),
        output_len=max_new_tokens,
        prompt_tokens=prompt_ids,
    )
    return scheduler.run_batch([req])[0].output_tokens


def run_e1(max_new_tokens: int = 16) -> bool:
    print("loading gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    cfg = model.config

    mgr = BlockManager(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=128,
        block_size=16,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = Scheduler(
        model, tokenizer, mgr,
        device="cpu",
        max_output_len=max_new_tokens,
        sample=False,
    )

    prompts = [
        "The capital of France is",
        "Once upon a time",
        "In the beginning, the universe",
        "1 + 1 =",
        "Roses are red,",
        "The quick brown fox jumps over the",
        "Hello",  # very short
        "To be, or not to be, that is the question whether tis nobler",  # longer
    ]

    n_pass = 0
    for i, prompt in enumerate(prompts):
        ref = reference_generate(model, tokenizer, prompt, max_new_tokens)
        paged = paged_generate(scheduler, tokenizer, prompt, max_new_tokens)
        match = ref == paged
        n_pass += int(match)
        status = "PASS" if match else "FAIL"
        print(f"[{status}] {prompt!r}")
        if not match:
            print(f"        ref tokens:   {ref}")
            print(f"        paged tokens: {paged}")
            print(f"        ref text:     {tokenizer.decode(ref)!r}")
            print(f"        paged text:   {tokenizer.decode(paged)!r}")
            # First divergence position
            for j in range(min(len(ref), len(paged))):
                if ref[j] != paged[j]:
                    print(f"        diverge at j={j}: ref={ref[j]} paged={paged[j]}")
                    break

    print(f"\n{n_pass}/{len(prompts)} prompts matched")
    print(f"blocks used after run: {mgr.n_used} / {mgr.n_blocks}  (expect 0)")
    return n_pass == len(prompts) and mgr.n_used == 0


if __name__ == "__main__":
    success = run_e1()
    sys.exit(0 if success else 1)
