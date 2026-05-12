"""Plot E5 sweep — v2 speedup vs N and vs prompt-length variance.

Two panels:
  Left  — speedup vs N (number of parallel requests) at fixed variance.
  Right — speedup vs prompt-length variance (lognormal sigma) at fixed N.

The single-point E5 result (1.28x at N=8, sigma=0.6) undersells continuous
batching. v2's win grows with both N (more sequences finish at staggered
times) and variance (larger gap between v1's max-padded prompt and v2's
sum-of-prompts compute).

Run from project root:
  python -m cpu.experiments.plot_e5_sweep
Output:
  plots/e5_speedup_sweep.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def workload(n: int, vocab_size: int, prompt_sigma: float, seed: int) -> List[Request]:
    rng = np.random.default_rng(seed)
    trng = torch.Generator(device="cpu").manual_seed(seed)
    prompt_mu, output_mu, output_sigma = 3.6, 2.7, 0.5
    reqs = []
    for i in range(n):
        plen = max(4, min(int(rng.lognormal(prompt_mu, prompt_sigma)), 128))
        olen = max(2, min(int(rng.lognormal(output_mu, output_sigma)), 40))
        tokens = torch.randint(0, vocab_size, (plen,), generator=trng).tolist()
        reqs.append(Request(
            request_id=f"r{i}",
            prompt_len=plen, output_len=olen, prompt_tokens=tokens,
        ))
    return reqs


def fresh(reqs):
    return [Request(
        request_id=r.request_id, prompt_len=r.prompt_len, output_len=r.output_len,
        arrival_time=r.arrival_time,
        prompt_tokens=list(r.prompt_tokens) if r.prompt_tokens else None,
    ) for r in reqs]


def _sync(device):
    if getattr(device, "type", device) == "cuda":
        torch.cuda.synchronize()


def measure(model, tokenizer, mgr_kwargs, reqs, max_olen, device="cpu"):
    """Return (v1_wall, v2_wall) for one workload."""
    warm = Request(request_id="w", prompt_len=8, output_len=2, prompt_tokens=[0]*8)

    mgr_v1 = BlockManager(**mgr_kwargs)
    sv1 = Scheduler(model, tokenizer, mgr_v1, device=device, max_output_len=max_olen)
    sv1.run_batch([Request(**vars(warm))])
    _sync(device)
    t0 = time.time(); sv1.run_batch(fresh(reqs)); _sync(device)
    v1 = time.time() - t0

    # free v1's block pool before allocating v2's — keeps the sweep within a
    # T4's memory when both managers are sized for the worst-case sequence.
    del sv1, mgr_v1
    if getattr(device, "type", device) == "cuda":
        torch.cuda.empty_cache()

    mgr_v2 = BlockManager(**mgr_kwargs)
    sv2 = ContinuousScheduler(model, tokenizer, mgr_v2, device=device,
                              n_slots=len(reqs), max_output_len=max_olen)
    sv2.run([Request(**vars(warm))])
    _sync(device)
    t0 = time.time(); sv2.run(fresh(reqs)); _sync(device)
    v2 = time.time() - t0
    return v1, v2


def main(model=None, tokenizer=None, device=None, dtype=None):
    """Generate plots/e5_speedup_sweep.png.

    Pass an already-loaded ``model``/``tokenizer`` (and ``device``/``dtype``)
    to drive the sweep on CUDA+fp16 from gpu.run_all; otherwise load GPT-2 on
    CPU in fp32. Unlike E2/E3, this is a wall-clock measurement, so the
    device it ran on matters and is recorded in the figure title.
    """
    if model is None:
        print("loading gpt2...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype
    cfg = model.config

    mgr_kwargs = dict(
        n_layers=cfg.n_layer, n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=4096, block_size=16,
        dtype=dtype, device=device,
    )

    # --- Sweep over N at fixed sigma=0.6 ---
    n_values = [4, 8, 12]
    n_sweep_speedup = []
    print("\nN sweep (sigma=0.6):")
    for n in n_values:
        reqs = workload(n, cfg.vocab_size, prompt_sigma=0.6, seed=0)
        max_olen = max(r.output_len for r in reqs)
        v1, v2 = measure(model, tokenizer, mgr_kwargs, reqs, max_olen, device=device)
        speedup = v1 / v2
        n_sweep_speedup.append(speedup)
        print(f"  N={n:>2}: v1={v1:.2f}s v2={v2:.2f}s speedup={speedup:.2f}x")

    # --- Sweep over sigma at fixed N=8 ---
    sigma_values = [0.3, 0.6, 0.9]
    sigma_sweep_speedup = []
    print("\nsigma sweep (N=8):")
    for sigma in sigma_values:
        reqs = workload(8, cfg.vocab_size, prompt_sigma=sigma, seed=0)
        plens = [r.prompt_len for r in reqs]
        max_olen = max(r.output_len for r in reqs)
        v1, v2 = measure(model, tokenizer, mgr_kwargs, reqs, max_olen, device=device)
        speedup = v1 / v2
        sigma_sweep_speedup.append(speedup)
        print(f"  sigma={sigma}: prompt_lens={plens} (max={max(plens)}) "
              f"v1={v1:.2f}s v2={v2:.2f}s speedup={speedup:.2f}x")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(n_values, n_sweep_speedup, "o-", color="tab:blue",
             linewidth=2, markersize=10)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax1.set_xlabel("N (parallel requests)")
    ax1.set_ylabel("v1 / v2 wallclock speedup")
    ax1.set_title("speedup vs batch size  (prompt sigma = 0.6)")
    ax1.set_xticks(n_values)
    ax1.grid(alpha=0.3)
    for x, y in zip(n_values, n_sweep_speedup):
        ax1.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center")

    ax2.plot(sigma_values, sigma_sweep_speedup, "s-", color="tab:orange",
             linewidth=2, markersize=10)
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("prompt-length sigma (lognormal)")
    ax2.set_ylabel("v1 / v2 wallclock speedup")
    ax2.set_title("speedup vs length variance  (N = 8)")
    ax2.set_xticks(sigma_values)
    ax2.grid(alpha=0.3)
    for x, y in zip(sigma_values, sigma_sweep_speedup):
        ax2.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center")

    dev = device.type if hasattr(device, "type") else str(device)
    dt = str(dtype).replace("torch.", "")
    fig.suptitle("E5: continuous batching speedup over uniform-batch v1 "
                 f"(GPT-2 small, {dev}/{dt})", y=1.02)
    fig.tight_layout()

    out = _ROOT / "plots" / "e5_speedup_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
