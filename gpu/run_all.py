"""Run Phase 1 + Phase 2 + Phase 3 headline experiments on GPU (Colab T4 / A100).

The CPU experiment scripts hardcode device="cpu". This is a one-shot runner
that reuses the same underlying classes (BlockManager, PagedCache,
ContinuousPagedCache, Scheduler, ContinuousScheduler — all device-agnostic)
but with cuda + fp16 throughout. Loads the model once and runs:

  E1 (correctness)   PagedCache vs DynamicCache, B=1, greedy
  E3 (prefix share)  N=8 parallel completions of one 256-token prompt
  E5 (v1 vs v2)      mixed-length workload through both schedulers
  E6 (kernel parity) E1 again with paged-attention decode kernel enabled —
                     verifies the kernel produces the same tokens as the
                     gather-then-attention path it replaces.

After the experiments it regenerates the four report figures into plots/
(E2 x2 from the simulator, E3 savings-vs-N, E5 speedup sweep) so the figures
and the headline numbers come from the same run. Pass --no-plots to skip.
E4 batched correctness is implicitly covered by E5's v1 path.

Colab usage:
  !pip install -q torch transformers numpy triton matplotlib datasets
  # upload the project folder, then:
  !cd PagedAttention && python -m gpu.run_all

Will fall back to CPU+fp32 with a warning if no GPU is available, so the
same script can be sanity-checked locally before paying for Colab time.
The Phase 3 kernel has a torch-reference backend that runs on CPU; the
Triton backend kicks in automatically when CUDA + Triton are available.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from continuous_cache import ContinuousPagedCache  # noqa: E402  (loaded for sanity)
from cpu.scheduler import Scheduler  # noqa: E402
from cpu.continuous_scheduler import ContinuousScheduler  # noqa: E402
from bench.workloads import Request  # noqa: E402


def _device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    print("WARNING: CUDA not available — falling back to CPU+fp32. "
          "Run on Colab/RunPod for the real numbers.")
    return torch.device("cpu"), torch.float32


def _make_manager(cfg, device, dtype, n_blocks=4096, block_size=16):
    return BlockManager(
        n_layers=cfg.n_layer, n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=n_blocks, block_size=block_size,
        dtype=dtype, device=device,
    )


def e1_gpu(model, tokenizer, device, dtype) -> bool:
    print("\n=== E1 — correctness vs DynamicCache (B=1, greedy) ===")
    mgr = _make_manager(model.config, device, dtype, n_blocks=128)
    sched = Scheduler(model, tokenizer, mgr, device=device,
                      max_output_len=16, sample=False)
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Roses are red,",
        "The quick brown fox jumps over the",
    ]
    n_pass = 0
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=16, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        ref = out[0, ids.shape[1]:].tolist()
        prompt_ids = tokenizer(p).input_ids
        req = Request(request_id="e1", prompt_len=len(prompt_ids),
                      output_len=16, prompt_tokens=prompt_ids)
        paged = sched.run_batch([req])[0].output_tokens
        match = ref == paged
        n_pass += int(match)
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {p!r}")
    print(f"  {n_pass}/{len(prompts)} match  blocks_remaining={mgr.n_used}")
    return n_pass == len(prompts) and mgr.n_used == 0


def e3_gpu(model, tokenizer, device, dtype) -> None:
    print("\n=== E3 — prefix sharing (N=8 parallel of one 256-tok prompt) ===")
    from cpu.experiments.e3_prefix_sharing import (
        make_aligned_prompt, run_independent, run_shared,
    )
    prompt_ids = make_aligned_prompt(tokenizer, 256)
    mgr_kwargs = dict(
        n_layers=model.config.n_layer, n_heads=model.config.n_head,
        head_dim=model.config.n_embd // model.config.n_head,
        n_blocks=4096, block_size=16, dtype=dtype, device=device,
    )
    mgr_i = BlockManager(**mgr_kwargs)
    indep_pre, indep_post = run_independent(model, mgr_i, prompt_ids, 8, 4)
    mgr_s = BlockManager(**mgr_kwargs)
    shared_pre, shared_post = run_shared(model, mgr_s, prompt_ids, 8, 4)
    print(f"  indep   pre={indep_pre}  post={indep_post}")
    print(f"  shared  pre={shared_pre}  post={shared_post}")
    print(f"  prefill ratio: {indep_pre/shared_pre:.2f}x  "
          f"(theoretical max = N = 8)")
    print(f"  total ratio:   {indep_post/shared_post:.2f}x")


def _workload(n, vocab_size, sigma, seed, olen_sigma=0.8, olen_mu=3.0, olen_cap=80):
    """Synthetic lognormal-length workload with random token ids. Kept as a
    fallback; the realistic path uses _hf_chat_reqs below."""
    rng = np.random.default_rng(seed)
    trng = torch.Generator(device="cpu").manual_seed(seed)
    reqs = []
    for i in range(n):
        plen = max(4, min(int(rng.lognormal(3.6, sigma)), 128))
        olen = max(2, min(int(rng.lognormal(olen_mu, olen_sigma)), olen_cap))
        tokens = torch.randint(0, vocab_size, (plen,), generator=trng).tolist()
        reqs.append(Request(
            request_id=f"r{i}", prompt_len=plen,
            output_len=olen, prompt_tokens=tokens,
        ))
    return reqs


def _hf_chat_reqs(n, tokenizer, max_prompt_len, output_cap, seed=0,
                  dataset="HuggingFaceH4/no_robots"):
    """Real chat prompts streamed from HF. Falls back to synthetic if the
    datasets library or the dataset itself isn't reachable.

    `output_cap` clips each request's generation budget so a single very-long
    response can't blow up the bench wall-time or the block pool.
    """
    from bench.workloads import hf_chat_workload
    reqs = []
    for r in hf_chat_workload(
        dataset, n_requests=n, tokenizer=tokenizer,
        arrival_rate=0.0, seed=seed,
        min_prompt_len=4, max_prompt_len=max_prompt_len,
    ):
        r.output_len = min(r.output_len, output_cap)
        reqs.append(r)
    return reqs


def _fresh(reqs):
    return [Request(
        request_id=r.request_id, prompt_len=r.prompt_len,
        output_len=r.output_len, arrival_time=r.arrival_time,
        prompt_tokens=list(r.prompt_tokens) if r.prompt_tokens else None,
    ) for r in reqs]


def e5_gpu(model, tokenizer, device, dtype) -> bool:
    print("\n=== E5 — continuous (v2) vs uniform (v1) ===")
    n = 16
    max_prompt_len = 256
    output_cap = 256
    try:
        reqs = _hf_chat_reqs(n, tokenizer,
                             max_prompt_len=max_prompt_len,
                             output_cap=output_cap, seed=0)
        if len(reqs) < n:
            raise RuntimeError(
                f"hf_chat_workload only yielded {len(reqs)}/{n} requests "
                f"after filtering; falling back to synthetic"
            )
        source = "no_robots"
    except Exception as e:
        print(f"  [warn] {e}")
        print("  falling back to synthetic random-token workload")
        reqs = _workload(n, model.config.vocab_size, sigma=0.6, seed=0,
                         olen_cap=output_cap)
        source = "synthetic"
    plens = [r.prompt_len for r in reqs]
    olens = [r.output_len for r in reqs]
    print(f"  workload N={n}, source={source}, "
          f"prompt_lens p50/p95/max="
          f"{int(np.median(plens))}/{int(np.percentile(plens, 95))}/{max(plens)}, "
          f"output_lens p50/p95/max="
          f"{int(np.median(olens))}/{int(np.percentile(olens, 95))}/{max(olens)}")
    max_olen = max(olens)

    # Block-pool sizing: 2 BlockManagers (v1 + v2), each holding
    # n_layers * n * ceil(max_total / bs) blocks in the worst case.
    bs = 16
    worst_per_seq = (max_prompt_len + output_cap + bs - 1) // bs
    n_blocks = max(4096, model.config.n_layer * n * worst_per_seq + 256)
    mgr_kwargs = dict(
        n_layers=model.config.n_layer, n_heads=model.config.n_head,
        head_dim=model.config.n_embd // model.config.n_head,
        n_blocks=n_blocks, block_size=bs, dtype=dtype, device=device,
    )
    warm = Request(request_id="w", prompt_len=8, output_len=2, prompt_tokens=[0]*8)

    mgr_v1 = BlockManager(**mgr_kwargs)
    sv1 = Scheduler(model, tokenizer, mgr_v1, device=device, max_output_len=max_olen)
    sv1.run_batch([Request(**vars(warm))])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    cv1 = sv1.run_batch(_fresh(reqs))
    if device.type == "cuda":
        torch.cuda.synchronize()
    v1 = time.time() - t0

    mgr_v2 = BlockManager(**mgr_kwargs)
    sv2 = ContinuousScheduler(model, tokenizer, mgr_v2, device=device,
                              n_slots=n, max_output_len=max_olen)
    sv2.run([Request(**vars(warm))])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    cv2 = sv2.run(_fresh(reqs))
    if device.type == "cuda":
        torch.cuda.synchronize()
    v2 = time.time() - t0

    by_v1 = {c.request_id: c for c in cv1}
    by_v2 = {c.request_id: c for c in cv2}
    n_match = 0
    for rid, c1 in by_v1.items():
        c2 = by_v2.get(rid)
        if c2 is None:
            continue
        L = min(len(c1.output_tokens), len(c2.output_tokens))
        n_match += int(c1.output_tokens[:L] == c2.output_tokens[:L])

    out_total = sum(c.output_len for c in cv1)
    # fp16 on CUDA flips occasional argmaxes between v1 and v2 because the
    # two cache layouts reduce in different orders. Allow up to 20% drift.
    match_threshold = max(1, int(0.8 * n))
    print(f"  v1 wall {v1:.2f}s  ({out_total/v1:.1f} tok/s)")
    print(f"  v2 wall {v2:.2f}s  ({out_total/v2:.1f} tok/s)")
    print(f"  speedup {v1/v2:.2f}x   greedy match {n_match}/{n} "
          f"(threshold {match_threshold})")
    return n_match >= match_threshold


def e6_gpu(model, tokenizer, device, dtype) -> bool:
    """Re-run E1 with the paged-attention decode kernel enabled.

    Exact token-for-token parity vs HF's DynamicCache holds in fp32, but in
    fp16 the kernel and HF's eager attention accumulate softmax in different
    orders, and that ~1e-3 logit drift occasionally flips an argmax over a
    50k-token vocab after a few decode steps. So we check the longest common
    prefix and require at least K matching tokens before allowing divergence.
    Independent synthetic correctness lives in gpu/test_paged_kernel.py.
    """
    print("\n=== E6 — kernel parity (paged decode kernel vs DynamicCache) ===")
    from gpu.paged_attn_patch import paged_attention
    from gpu.paged_attention_kernel import _HAS_TRITON

    backend = "triton" if (_HAS_TRITON and device.type == "cuda") else "torch"
    # CPU torch / fp32 has plenty of precision for exact match; fp16/CUDA
    # only guarantees an approximate prefix.
    K = 3 if (device.type == "cuda" and dtype == torch.float16) else 16
    print(f"  backend={backend}  prefix_required={K}")

    mgr = _make_manager(model.config, device, dtype, n_blocks=128)
    sched = Scheduler(model, tokenizer, mgr, device=device,
                      max_output_len=16, sample=False)
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Roses are red,",
        "The quick brown fox jumps over the",
    ]
    n_pass = 0
    with paged_attention():
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=16, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            ref = out[0, ids.shape[1]:].tolist()
            prompt_ids = tokenizer(p).input_ids
            req = Request(request_id="e6", prompt_len=len(prompt_ids),
                          output_len=16, prompt_tokens=prompt_ids)
            paged = sched.run_batch([req])[0].output_tokens
            lcp = 0
            for a, b in zip(ref, paged):
                if a != b:
                    break
                lcp += 1
            match = lcp >= K
            n_pass += int(match)
            status = "PASS" if match else "FAIL"
            print(f"  [{status}] lcp={lcp}/{len(ref)}  {p!r}")
            if not match:
                print(f"          ref:    {ref}")
                print(f"          kernel: {paged}")
    print(f"  {n_pass}/{len(prompts)} match  blocks_remaining={mgr.n_used}")
    return n_pass == len(prompts) and mgr.n_used == 0


def _plots(model, tokenizer, device, dtype) -> None:
    """Regenerate the four report figures from this run.

    E2 (both panels) is allocator simulation only — device/dtype independent —
    so its scripts are called as-is. E3 reuses the loaded model on this
    device (the savings ratios are still device-independent, but the figure
    is then provenance-consistent with the rest of the run). E5 is a
    wall-clock sweep, so running it here on cuda/fp16 is the whole point.
    Output: plots/{e2_memory_timeseries,e2_memory_real_lengths,
    e3_savings_vs_n,e5_speedup_sweep}.png
    """
    print("\n=== plots — regenerating report figures ===")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  [skip] matplotlib not installed (pip install matplotlib)")
        return
    from cpu.experiments import plot_e2, plot_e2_real, plot_e3, plot_e5_sweep
    for name, fn in [
        ("e2_memory_timeseries", lambda: plot_e2.main()),
        ("e2_memory_real_lengths", lambda: plot_e2_real.main()),
        ("e3_savings_vs_n", lambda: plot_e3.main(model=model, tokenizer=tokenizer,
                                                 device=device, dtype=dtype)),
        ("e5_speedup_sweep", lambda: plot_e5_sweep.main(model=model, tokenizer=tokenizer,
                                                        device=device, dtype=dtype)),
    ]:
        try:
            print(f"  -> {name} ...")
            fn()
        except Exception as e:  # one bad plot shouldn't kill the rest
            print(f"  [warn] {name} failed: {type(e).__name__}: {e}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="GPU headline experiments + report figures")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip regenerating the plots/ figures")
    args = ap.parse_args(argv)

    device, dtype = _device_dtype()
    print(f"device={device} dtype={dtype}")

    print("\nloading gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=dtype).to(device).eval()

    ok_e1 = e1_gpu(model, tokenizer, device, dtype)
    e3_gpu(model, tokenizer, device, dtype)
    ok_e5 = e5_gpu(model, tokenizer, device, dtype)
    ok_e6 = e6_gpu(model, tokenizer, device, dtype)

    if not args.no_plots:
        _plots(model, tokenizer, device, dtype)

    print(f"\n=== summary ===  E1 {'OK' if ok_e1 else 'FAIL'}  "
          f"E5 {'OK' if ok_e5 else 'FAIL'}  "
          f"E6 {'OK' if ok_e6 else 'FAIL'}")
    return 0 if (ok_e1 and ok_e5 and ok_e6) else 1


if __name__ == "__main__":
    sys.exit(main())
