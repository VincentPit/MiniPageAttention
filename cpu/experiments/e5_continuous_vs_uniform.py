"""E5 — Continuous batching (v2) vs uniform batch (v1).

Same workload, same model, two schedulers. Two sources of v2 advantage:

  1. Prefill compute. v1 pads all prompts to max_prompt_len and runs them
     as one B=N forward, doing work proportional to B * max_prompt_len.
     v2 runs N separate B=1 prefills, doing work proportional to sum(prompt_len).
     Lossier the variance in prompt_len, the bigger the v2 win.

  2. Decode compute. v1 keeps every sequence in the batch until *all* are
     done — finished sequences keep getting model() calls whose outputs
     are discarded. v2 removes them as they finish, so the active batch
     shrinks naturally.

Greedy decoding so v1 and v2 outputs should be byte-identical.

Run from project root:
  python -m cpu.experiments.e5_continuous_vs_uniform
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from cpu.scheduler import Scheduler  # noqa: E402
from cpu.continuous_scheduler import ContinuousScheduler  # noqa: E402
from bench.workloads import Request  # noqa: E402


def make_workload(n_requests: int, vocab_size: int, seed: int = 0) -> List[Request]:
    """Mixed-length synthetic workload with pre-generated prompt tokens.

    Heavy variance is what gives v2 something to optimize away — fixing the
    seed lets us point at the same workload across schedulers.
    """
    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator(device="cpu").manual_seed(seed)

    # Lognormal lengths capped to keep CPU runtime tractable.
    prompt_mu, prompt_sigma = 3.6, 0.6  # median ~37, p95 ~100
    output_mu, output_sigma = 2.7, 0.5  # median ~15, p95 ~34

    reqs: List[Request] = []
    for i in range(n_requests):
        plen = max(4, min(int(rng.lognormal(prompt_mu, prompt_sigma)), 128))
        olen = max(2, min(int(rng.lognormal(output_mu, output_sigma)), 40))
        tokens = torch.randint(0, vocab_size, (plen,), generator=torch_rng).tolist()
        reqs.append(Request(
            request_id=f"r{i}",
            prompt_len=plen,
            output_len=olen,
            prompt_tokens=tokens,
        ))
    return reqs


def fresh_copies(reqs: List[Request]) -> List[Request]:
    return [
        Request(
            request_id=r.request_id,
            prompt_len=r.prompt_len,
            output_len=r.output_len,
            arrival_time=r.arrival_time,
            prompt_tokens=list(r.prompt_tokens) if r.prompt_tokens else None,
        )
        for r in reqs
    ]


def main():
    print("loading gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    cfg = model.config

    n_requests = 8
    requests = make_workload(n_requests, cfg.vocab_size, seed=0)

    plens = [r.prompt_len for r in requests]
    olens = [r.output_len for r in requests]
    print(f"\nworkload: {n_requests} requests")
    print(f"  prompt_lens: {plens}")
    print(f"  output_lens: {olens}")
    print(f"  prompt total {sum(plens)}, max {max(plens)}, "
          f"v1 prefill compute proportional to {n_requests * max(plens)}")
    print(f"  output total {sum(olens)}, max {max(olens)}, "
          f"v1 decode compute proportional to {n_requests * max(olens)}")

    block_size = 16
    mgr_kwargs = dict(
        n_layers=cfg.n_layer, n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=2048, block_size=block_size,
        dtype=torch.float32, device="cpu",
    )

    # Warmup request shared between configs
    warm = Request(request_id="warm", prompt_len=8, output_len=2,
                   prompt_tokens=[0] * 8)

    # --- v1 Scheduler ---
    mgr_v1 = BlockManager(**mgr_kwargs)
    sched_v1 = Scheduler(model, tokenizer, mgr_v1, max_output_len=max(olens))
    sched_v1.run_batch([Request(**vars(warm))])  # warmup
    reqs_v1 = fresh_copies(requests)
    t0 = time.time()
    completions_v1 = sched_v1.run_batch(reqs_v1)
    v1_wall = time.time() - t0

    # --- v2 ContinuousScheduler ---
    mgr_v2 = BlockManager(**mgr_kwargs)
    sched_v2 = ContinuousScheduler(model, tokenizer, mgr_v2,
                                   n_slots=n_requests,
                                   max_output_len=max(olens))
    sched_v2.run([Request(**vars(warm))])  # warmup
    reqs_v2 = fresh_copies(requests)
    t0 = time.time()
    completions_v2 = sched_v2.run(reqs_v2)
    v2_wall = time.time() - t0

    # --- Cross-check: greedy outputs should match per request ---
    by_id_v1 = {c.request_id: c for c in completions_v1}
    by_id_v2 = {c.request_id: c for c in completions_v2}
    n_match = 0
    mismatches = []
    for rid, c1 in by_id_v1.items():
        c2 = by_id_v2.get(rid)
        if c2 is None:
            continue
        L = min(len(c1.output_tokens), len(c2.output_tokens))
        if c1.output_tokens[:L] == c2.output_tokens[:L]:
            n_match += 1
        else:
            mismatches.append((rid, c1.output_tokens, c2.output_tokens))

    out_v1 = sum(c.output_len for c in completions_v1)
    out_v2 = sum(c.output_len for c in completions_v2)

    print(f"\n  {'':<22} {'v1 (uniform)':>14} {'v2 (continuous)':>16} {'v1/v2':>8}")
    print(f"  {'wall (s)':<22} {v1_wall:>14.2f} {v2_wall:>16.2f} "
          f"{v1_wall/v2_wall:>7.2f}x")
    print(f"  {'output tokens':<22} {out_v1:>14} {out_v2:>16}")
    print(f"  {'tokens/s':<22} {out_v1/v1_wall:>14.2f} {out_v2/v2_wall:>16.2f}")
    p50_v1 = float(np.percentile([c.latency for c in completions_v1], 50))
    p50_v2 = float(np.percentile([c.latency for c in completions_v2], 50))
    p95_v1 = float(np.percentile([c.latency for c in completions_v1], 95))
    p95_v2 = float(np.percentile([c.latency for c in completions_v2], 95))
    print(f"  {'latency p50 (s)':<22} {p50_v1:>14.2f} {p50_v2:>16.2f}")
    print(f"  {'latency p95 (s)':<22} {p95_v1:>14.2f} {p95_v2:>16.2f}")

    print(f"\n  greedy outputs match:  {n_match}/{len(requests)}")
    if mismatches:
        rid, t1, t2 = mismatches[0]
        L = min(len(t1), len(t2))
        for j in range(L):
            if t1[j] != t2[j]:
                print(f"    first diff in {rid} at j={j}: v1={t1[j]} v2={t2[j]}")
                break
    print(f"  blocks remaining: v1={mgr_v1.n_used}, v2={mgr_v2.n_used}")
    return n_match == len(requests) and mgr_v1.n_used == 0 and mgr_v2.n_used == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
