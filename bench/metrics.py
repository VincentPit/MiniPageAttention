"""Measurement helpers: allocation traces, latency stats, no-model simulator.

The simulator is the workhorse for E2 (fragmentation): it walks a workload
through a discrete-event model of allocate/free operations without running
any actual model, so a 10K-request sweep takes seconds instead of hours.

For E2 with a real model, hook into BlockManager.allocate/free directly
and feed events into AllocationTrace.record().
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np

from .workloads import Request


@dataclass
class AllocationTrace:
    """Time-series of block-pool occupancy.

    events: (time, n_blocks_in_use, event_type, request_id)
    event_type in {"alloc", "free"}.
    """
    events: List[Tuple[float, int, str, str]] = field(default_factory=list)

    def record(self, time: float, n_used: int, event_type: str, request_id: str) -> None:
        self.events.append((time, n_used, event_type, request_id))

    @property
    def peak(self) -> int:
        return max((e[1] for e in self.events), default=0)

    @property
    def mean(self) -> float:
        """Time-weighted mean occupancy."""
        if len(self.events) < 2:
            return float(self.events[0][1]) if self.events else 0.0
        total_time = self.events[-1][0] - self.events[0][0]
        if total_time == 0:
            return float(self.events[-1][1])
        weighted = sum(
            self.events[i][1] * (self.events[i + 1][0] - self.events[i][0])
            for i in range(len(self.events) - 1)
        )
        return weighted / total_time

    def utilization(self, pool_size: int) -> float:
        return self.peak / pool_size if pool_size > 0 else 0.0


@dataclass
class LatencyStats:
    p50: float
    p95: float
    p99: float
    mean: float
    n: int

    @classmethod
    def from_latencies(cls, latencies: Iterable[float]) -> "LatencyStats":
        arr = np.array(list(latencies), dtype=float)
        if len(arr) == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0)
        return cls(
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            mean=float(arr.mean()),
            n=len(arr),
        )

    def __repr__(self) -> str:
        return (f"LatencyStats(n={self.n}, p50={self.p50:.3f}, "
                f"p95={self.p95:.3f}, p99={self.p99:.3f}, mean={self.mean:.3f})")


def simulate_allocator(
    requests: Iterable[Request],
    block_size: int,
    decode_step_time: float = 0.01,
    prefill_time: float = 0.05,
    naive: bool = False,
    max_seq_len: int = 2048,
) -> AllocationTrace:
    """Discrete-event simulation of block allocation through a workload.

    Each request:
      1. At arrival_time + prefill_time:
           paged: allocates ceil(prompt_len / block_size) blocks
           naive: allocates ceil(max_seq_len / block_size) blocks (worst-case)
      2. Every decode_step_time: extends by one token, allocating a fresh
         block on each block_size boundary (paged only — naive pre-allocated)
      3. At completion: frees all its blocks

    Run twice (naive=False, then True) to get the headline E2 ratio.
    """
    requests = sorted(requests, key=lambda r: r.arrival_time)
    naive_blocks = math.ceil(max_seq_len / block_size)

    heap: list = []
    state = {}
    counter = 0

    for i, req in enumerate(requests):
        heapq.heappush(heap, (req.arrival_time + prefill_time, i, i, "start"))
        state[i] = {"tokens_done": 0, "blocks_held": 0, "req": req}

    trace = AllocationTrace()
    in_use = 0

    while heap:
        t, _, ridx, kind = heapq.heappop(heap)
        s = state[ridx]
        req = s["req"]

        if kind == "start":
            need = naive_blocks if naive else math.ceil(req.prompt_len / block_size)
            in_use += need
            s["blocks_held"] = need
            s["tokens_done"] = req.prompt_len
            trace.record(t, in_use, "alloc", req.request_id)
            counter += 1
            heapq.heappush(heap, (t + decode_step_time, counter, ridx, "step"))

        elif kind == "step":
            s["tokens_done"] += 1
            if not naive:
                need = math.ceil(s["tokens_done"] / block_size)
                if need > s["blocks_held"]:
                    in_use += (need - s["blocks_held"])
                    s["blocks_held"] = need
                    trace.record(t, in_use, "alloc", req.request_id)

            if s["tokens_done"] >= req.prompt_len + req.output_len:
                in_use -= s["blocks_held"]
                trace.record(t, in_use, "free", req.request_id)
                del state[ridx]
            else:
                counter += 1
                heapq.heappush(heap, (t + decode_step_time, counter, ridx, "step"))

    return trace


def peak_concurrency(
    requests: Iterable[Request],
    decode_step_time: float = 0.01,
    prefill_time: float = 0.05,
) -> int:
    """Max number of simultaneously-active requests in the workload."""
    events = []
    for req in requests:
        start = req.arrival_time + prefill_time
        end = start + (req.output_len * decode_step_time)
        events.append((start, +1))
        events.append((end, -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


if __name__ == "__main__":
    # Demo: paged vs naive on a synthetic workload.
    from .workloads import synthetic_workload

    reqs = list(synthetic_workload(n_requests=2000, arrival_rate=20.0, seed=0))
    block_size = 16

    paged = simulate_allocator(reqs, block_size=block_size, naive=False)
    naive = simulate_allocator(reqs, block_size=block_size, naive=True, max_seq_len=2048)

    pc = peak_concurrency(reqs)
    ratio = naive.peak / paged.peak if paged.peak else float("inf")
    print(f"requests:           {len(reqs)}")
    print(f"peak concurrent:    {pc}")
    print(f"paged peak blocks:  {paged.peak}")
    print(f"paged mean blocks:  {paged.mean:.1f}")
    print(f"naive peak blocks:  {naive.peak}")
    print(f"naive / paged:      {ratio:.2f}x")
