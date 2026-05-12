"""Plot E2 with real conversation lengths (HuggingFaceH4/no_robots).

Streams ~1500 examples from a public conversational dataset, tokenizes
each prompt+response with GPT-2, then runs the paged and naive simulators
on those lengths under a Poisson arrival pattern. Produces:
  plots/e2_memory_real_lengths.png

For the actual ShareGPT dataset (700MB JSON) use sharegpt_workload with a
local path; this script uses no_robots because it streams.

Run from project root:
  python -m cpu.experiments.plot_e2_real
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.metrics import peak_concurrency, simulate_allocator  # noqa: E402
from bench.workloads import hf_chat_workload  # noqa: E402


def trace_xy(trace):
    return [e[0] for e in trace.events], [e[1] for e in trace.events]


def reassign_arrivals(reqs, arrival_rate: float, seed: int = 0):
    """Replay tokenized requests under a Poisson arrival pattern."""
    rng = np.random.default_rng(seed)
    t = 0.0
    out = []
    for r in reqs:
        t += rng.exponential(1.0 / arrival_rate)
        r.arrival_time = t
        out.append(r)
    return out


def main():
    n_requests = 1500
    arrival_rate = 20.0
    block_size = 16
    max_seq_len = 2048

    print("loading tokenizer + streaming no_robots...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    t_load = time.time()
    reqs = list(hf_chat_workload(
        "HuggingFaceH4/no_robots", n_requests=n_requests,
        tokenizer=tokenizer, arrival_rate=0, seed=0,
        max_prompt_len=max_seq_len,
    ))
    print(f"loaded {len(reqs)} requests in {time.time()-t_load:.1f}s")
    plens = [r.prompt_len for r in reqs]
    olens = [r.output_len for r in reqs]
    print(f"  prompt_len   median={np.median(plens):.0f}  p95={np.percentile(plens,95):.0f}  max={max(plens)}")
    print(f"  output_len   median={np.median(olens):.0f}  p95={np.percentile(olens,95):.0f}  max={max(olens)}")

    reqs = reassign_arrivals(reqs, arrival_rate=arrival_rate, seed=42)

    paged = simulate_allocator(reqs, block_size=block_size, naive=False)
    naive = simulate_allocator(
        reqs, block_size=block_size, naive=True, max_seq_len=max_seq_len,
    )
    pc = peak_concurrency(reqs)

    px, py = trace_xy(paged)
    nx, ny = trace_xy(naive)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.step(nx, ny, where="post", color="tab:red", alpha=0.85,
            label=f"naive (peak={naive.peak})")
    ax.step(px, py, where="post", color="tab:blue", alpha=0.85,
            label=f"paged (peak={paged.peak})")
    ax.axhline(naive.peak, color="tab:red", linestyle=":", alpha=0.4)
    ax.axhline(paged.peak, color="tab:blue", linestyle=":", alpha=0.4)

    ratio = naive.peak / paged.peak
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("blocks in use (per layer)")
    ax.set_title(
        f"E2 (real lengths): block-pool occupancy on HuggingFaceH4/no_robots\n"
        f"{len(reqs)} requests, {arrival_rate} qps, peak concurrency {pc}, "
        f"naive/paged = {ratio:.2f}x"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    out = _ROOT / "plots" / "e2_memory_real_lengths.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out.relative_to(_ROOT)}  (paged peak {paged.peak}, "
          f"naive peak {naive.peak}, ratio {ratio:.2f}x)")


if __name__ == "__main__":
    main()
