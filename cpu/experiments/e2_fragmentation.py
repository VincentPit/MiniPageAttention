"""E2 — Fragmentation.

Two parts:

  1. Validation
     Run a small fixed-length batch through the real cache + scheduler with
     the BlockManager allocation hooks attached. Compare measured peak block
     usage to the simulator's prediction (multiplied by n_layers, since the
     simulator counts per-sequence blocks and the manager allocates a fresh
     block per layer per boundary). If they agree, the simulator is a faithful
     stand-in for headline measurements that don't fit on a CPU box.

  2. Headline (simulator only)
     Synthetic 2K-request lognormal workload through the simulator. Compare
     paged peak vs naive (every active request reserves max_seq_len blocks).
     This is the same primitive as the bench/__main__ smoke test, run here
     in the context of a Project A Phase 1 deliverable.

Run from project root:
  python -m cpu.experiments.e2_fragmentation
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from cpu.scheduler import Scheduler  # noqa: E402
from bench.workloads import fixed_length_workload, synthetic_workload  # noqa: E402
from bench.metrics import (  # noqa: E402
    AllocationTrace,
    peak_concurrency,
    simulate_allocator,
)


def attach_trace(mgr: BlockManager) -> AllocationTrace:
    """Wire BlockManager hooks into a new AllocationTrace."""
    trace = AllocationTrace()
    t0 = time.time()
    mgr.on_alloc = lambda bid, n_used: trace.record(
        time.time() - t0, n_used, "alloc", str(bid),
    )
    mgr.on_free = lambda bid, n_used: trace.record(
        time.time() - t0, n_used, "free", str(bid),
    )
    return trace


def part_validation() -> bool:
    """Real-cache peak == simulator peak * n_layers (uniform-length workload)."""
    print("=== Part 1: Real-cache validation ===")

    print("loading gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    cfg = model.config

    n_layers = cfg.n_layer
    block_size = 16
    n_requests = 4
    prompt_len = 32
    output_len = 8

    mgr = BlockManager(
        n_layers=n_layers,
        n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=4096,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )
    trace = attach_trace(mgr)

    sched = Scheduler(model, tokenizer, mgr, device="cpu", max_output_len=output_len)
    reqs = list(fixed_length_workload(
        n_requests=n_requests, prompt_len=prompt_len, output_len=output_len,
    ))
    sched.run_batch(reqs)

    measured_peak = trace.peak

    sim_reqs = list(fixed_length_workload(
        n_requests=n_requests, prompt_len=prompt_len, output_len=output_len,
    ))
    sim_peak = simulate_allocator(sim_reqs, block_size=block_size, naive=False).peak
    expected_real = sim_peak * n_layers

    print(f"  N={n_requests}, prompt_len={prompt_len}, output_len={output_len}, "
          f"layers={n_layers}, block_size={block_size}")
    print(f"  measured peak (real cache):           {measured_peak}")
    print(f"  simulator peak (per layer):           {sim_peak}")
    print(f"  simulator * n_layers (expected real): {expected_real}")
    print(f"  match:                                {measured_peak == expected_real}")
    print(f"  blocks remaining after run:           {mgr.n_used}  (expect 0)")

    return measured_peak == expected_real and mgr.n_used == 0


def part_headline() -> None:
    """Headline E2: paged vs naive on a synthetic varied-length workload."""
    print("\n=== Part 2: Headline simulator sweep ===")

    n_requests = 2000
    arrival_rate = 20.0
    block_size = 16
    max_seq_len = 2048

    reqs = list(synthetic_workload(
        n_requests=n_requests, arrival_rate=arrival_rate, seed=0,
    ))

    paged = simulate_allocator(reqs, block_size=block_size, naive=False)
    naive = simulate_allocator(
        reqs, block_size=block_size, naive=True, max_seq_len=max_seq_len,
    )
    pc = peak_concurrency(reqs)

    n_layers_gpt2 = 12
    bytes_per_layer_block = 2 * block_size * 12 * 64 * 4  # K+V, GPT-2 small fp32
    bytes_per_block = bytes_per_layer_block * n_layers_gpt2

    print(f"  N={n_requests}, arrivals={arrival_rate} qps, block_size={block_size}")
    print(f"  peak concurrency (active requests at peak): {pc}")
    print(f"  {'':<24} {'per-layer':>12} {'across-12L MB':>15}")
    print(f"  {'paged peak':<24} {paged.peak:>12} "
          f"{paged.peak * bytes_per_block / 1e6:>15.1f}")
    print(f"  {'naive peak':<24} {naive.peak:>12} "
          f"{naive.peak * bytes_per_block / 1e6:>15.1f}")
    print(f"  {'paged time-mean':<24} {paged.mean:>12.1f} "
          f"{paged.mean * bytes_per_block / 1e6:>15.1f}")
    print(f"\n  naive / paged (peak): {naive.peak / paged.peak:.2f}x")


if __name__ == "__main__":
    ok = part_validation()
    part_headline()
    sys.exit(0 if ok else 1)
