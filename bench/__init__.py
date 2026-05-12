"""Workload generation and measurement harness for pagedkv experiments.

Backend-agnostic. Generates Request streams that can be fed to:
  - the CPU PagedCache + a real model (E1/E3/E4)
  - the GPU PagedCache + a real model (same)
  - the no-model allocator simulator (E2 — fast, no GPU required)
"""
from .workloads import (
    Request,
    fixed_length_workload,
    hf_chat_workload,
    shared_prefix_workload,
    sharegpt_workload,
    synthetic_workload,
)
from .metrics import (
    AllocationTrace,
    LatencyStats,
    peak_concurrency,
    simulate_allocator,
)

__all__ = [
    "Request",
    "fixed_length_workload",
    "hf_chat_workload",
    "shared_prefix_workload",
    "sharegpt_workload",
    "synthetic_workload",
    "AllocationTrace",
    "LatencyStats",
    "peak_concurrency",
    "simulate_allocator",
]
