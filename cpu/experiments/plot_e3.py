"""Plot E3 — savings ratio vs N parallel.

Sweeps N over [2, 4, 8, 16] and measures (independent / shared) block-pool
peak through the real cache. Two curves: ratio after prefill (theoretical
max, = N) and ratio after a few decode steps (decays toward 1 as decode
grows because output blocks are per-sequence).

Run from project root:
  python -m cpu.experiments.plot_e3
Output:
  plots/e3_savings_vs_n.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from cpu.experiments.e3_prefix_sharing import (  # noqa: E402
    make_aligned_prompt,
    run_independent,
    run_shared,
)


def main(model=None, tokenizer=None, device=None, dtype=None):
    """Generate plots/e3_savings_vs_n.png.

    If ``model``/``tokenizer`` are passed in, reuse them (so gpu.run_all can
    drive this on CUDA+fp16 without reloading); otherwise load GPT-2 on CPU
    in fp32. The savings ratios are block counts and are device/dtype
    independent, but the run that produced the figure is recorded in the
    title for provenance.
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

    prompt_len = 256
    decode_steps = 4
    block_size = 16
    n_values = [2, 4, 8, 16]

    prompt_ids = make_aligned_prompt(tokenizer, prompt_len)

    mgr_kwargs = dict(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=8192,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    prefill_ratios = []
    total_ratios = []
    for N in n_values:
        mgr_i = BlockManager(**mgr_kwargs)
        indep_pre, indep_post = run_independent(model, mgr_i, prompt_ids, N, decode_steps)
        mgr_s = BlockManager(**mgr_kwargs)
        shared_pre, shared_post = run_shared(model, mgr_s, prompt_ids, N, decode_steps)
        prefill_ratios.append(indep_pre / shared_pre)
        total_ratios.append(indep_post / shared_post)
        print(f"  N={N:>3}: prefill {indep_pre/shared_pre:.2f}x, "
              f"after {decode_steps} decode {indep_post/shared_post:.2f}x")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(n_values, n_values, "k--", alpha=0.4, label="theoretical max (= N)")
    ax.plot(n_values, prefill_ratios, "o-", color="tab:blue",
            label="after prefill", linewidth=2, markersize=8)
    ax.plot(n_values, total_ratios, "s-", color="tab:orange",
            label=f"after {decode_steps} decode steps", linewidth=2, markersize=8)
    ax.set_xlabel("N (parallel completions)")
    ax.set_ylabel("memory savings ratio (independent / shared)")
    dt = str(dtype).replace("torch.", "")
    ax.set_title(
        f"E3: prefix-sharing savings vs N\n"
        f"prompt_len={prompt_len}, decode_steps={decode_steps}, "
        f"GPT-2 small ({device.type if hasattr(device, 'type') else device}/{dt})"
    )
    ax.set_xticks(n_values)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    out = _ROOT / "plots" / "e3_savings_vs_n.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
