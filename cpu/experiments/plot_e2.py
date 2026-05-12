"""Plot E2 — memory-vs-time time series.

Runs paged and naive simulators on a synthetic ShareGPT-shaped workload and
saves a step plot of block-pool occupancy over time. No model is loaded.

Run from project root:
  python -m cpu.experiments.plot_e2
Output:
  plots/e2_memory_timeseries.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.metrics import simulate_allocator  # noqa: E402
from bench.workloads import synthetic_workload  # noqa: E402


def trace_xy(trace):
    """Return (times, counts) lists from an AllocationTrace."""
    return [e[0] for e in trace.events], [e[1] for e in trace.events]


def main():
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

    px, py = trace_xy(paged)
    nx, ny = trace_xy(naive)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.step(nx, ny, where="post", label=f"naive (peak={naive.peak})",
            color="tab:red", alpha=0.85)
    ax.step(px, py, where="post", label=f"paged (peak={paged.peak})",
            color="tab:blue", alpha=0.85)
    ax.axhline(naive.peak, color="tab:red", linestyle=":", alpha=0.4)
    ax.axhline(paged.peak, color="tab:blue", linestyle=":", alpha=0.4)

    ratio = naive.peak / paged.peak
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("blocks in use (per layer)")
    ax.set_title(
        f"E2: block-pool occupancy under a synthetic ShareGPT-shaped workload\n"
        f"{n_requests} requests, {arrival_rate} qps, "
        f"naive/paged peak = {ratio:.2f}x"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    out = _ROOT / "plots" / "e2_memory_timeseries.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out.relative_to(_ROOT)}  "
          f"(paged peak {paged.peak}, naive peak {naive.peak}, ratio {ratio:.2f}x)")


if __name__ == "__main__":
    main()
