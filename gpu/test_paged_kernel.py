"""Synthetic correctness harness for paged_attention_decode.

Builds a paged KV pool with random blocks and per-sequence block tables of
variable lengths, then compares:
  - paged_attention_decode_torch  (paged, no gather of full sequence)
  - naive_attention_for_test      (gathered + masked reference)
  - paged_attention_decode_triton (if CUDA + Triton available)

All three must agree to high precision in fp32 and reasonable precision in
fp16. Run from project root:

  python -m gpu.test_paged_kernel
"""
from __future__ import annotations

import sys

import torch

from gpu.paged_attention_kernel import (
    paged_attention_decode_torch,
    naive_attention_for_test,
    _HAS_TRITON,
)


def _build_paged_scenario(B, H, D, block_size, n_blocks, max_len, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    seq_lens = torch.randint(1, max_len + 1, (B,), generator=g).to(torch.int32)

    # Allocate distinct blocks for each (sequence, block_idx).
    max_blocks = (int(seq_lens.max().item()) + block_size - 1) // block_size
    block_tables = torch.full((B, max_blocks), -1, dtype=torch.int32)
    bid_iter = iter(range(n_blocks))
    for b in range(B):
        L = int(seq_lens[b].item())
        n_b = (L + block_size - 1) // block_size
        for j in range(n_b):
            block_tables[b, j] = next(bid_iter)

    k_pool = torch.randn(n_blocks, block_size, H, D, generator=g, dtype=torch.float32).to(dtype=dtype, device=device)
    v_pool = torch.randn(n_blocks, block_size, H, D, generator=g, dtype=torch.float32).to(dtype=dtype, device=device)
    query = torch.randn(B, H, D, generator=g, dtype=torch.float32).to(dtype=dtype, device=device)

    block_tables = block_tables.to(device)
    seq_lens = seq_lens.to(device)

    # Build the gathered reference: [B, max_L, H, D] with zero padding.
    max_L = int(seq_lens.max().item())
    K_full = torch.zeros(B, max_L, H, D, dtype=dtype, device=device)
    V_full = torch.zeros(B, max_L, H, D, dtype=dtype, device=device)
    for b in range(B):
        L = int(seq_lens[b].item())
        n_b = (L + block_size - 1) // block_size
        bids = block_tables[b, :n_b].to(torch.long)
        Kb = k_pool[bids].reshape(-1, H, D)[:L]
        Vb = v_pool[bids].reshape(-1, H, D)[:L]
        K_full[b, :L] = Kb
        V_full[b, :L] = Vb

    return query, k_pool, v_pool, block_tables, seq_lens, K_full, V_full


def _run(name: str, dtype, device, atol_paged, atol_triton):
    print(f"\n--- {name}  device={device}  dtype={dtype} ---")
    B, H, D = 3, 4, 64
    block_size = 16
    n_blocks = 64
    max_len = 90  # spans up to 6 blocks
    scale = D ** -0.5

    q, k_pool, v_pool, bt, sl, K_full, V_full = _build_paged_scenario(
        B, H, D, block_size, n_blocks, max_len, dtype, device,
    )

    ref = naive_attention_for_test(q, K_full, V_full, sl, scale)
    paged = paged_attention_decode_torch(q, k_pool, v_pool, bt, sl, scale)

    diff_paged = (ref - paged).abs().max().item()
    print(f"  paged-torch vs naive   max|diff| = {diff_paged:.2e}  "
          f"(tol {atol_paged:.0e})")
    ok = diff_paged < atol_paged

    if _HAS_TRITON and device.type == "cuda":
        from gpu.paged_attention_kernel import paged_attention_decode_triton
        triton_out = paged_attention_decode_triton(q, k_pool, v_pool, bt, sl, scale)
        diff_tri = (ref - triton_out).abs().max().item()
        print(f"  paged-triton vs naive  max|diff| = {diff_tri:.2e}  "
              f"(tol {atol_triton:.0e})")
        ok = ok and diff_tri < atol_triton
    else:
        print(f"  paged-triton: skipped (triton={_HAS_TRITON}, device={device.type})")

    return ok


def main():
    cpu = torch.device("cpu")
    ok_fp32 = _run("fp32 / cpu", torch.float32, cpu, atol_paged=1e-5, atol_triton=1e-5)

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        ok_fp16 = _run("fp16 / cuda", torch.float16, cuda, atol_paged=5e-3, atol_triton=5e-3)
    else:
        print("\n(no CUDA — skipping fp16/cuda)")
        ok_fp16 = True

    print(f"\n=== {'PASS' if (ok_fp32 and ok_fp16) else 'FAIL'} ===")
    return 0 if (ok_fp32 and ok_fp16) else 1


if __name__ == "__main__":
    sys.exit(main())
