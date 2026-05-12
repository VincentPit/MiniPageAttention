"""Workload generators: streams of Request objects.

Four workloads cover the experiments in the project plan:

  synthetic_workload   — lognormal lengths + Poisson arrivals (E2 fast path, E4)
  sharegpt_workload    — replay real conversations (E2/E4 credibility)
  shared_prefix_workload — N requests sharing one prompt (E3 prefix sharing)
  fixed_length_workload — uniform lengths (E1 correctness, ablations)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np


@dataclass
class Request:
    """A single inference request.

    For allocator-only experiments (E2) only lengths are needed and
    `prompt_tokens` may be None. For experiments that run a real model,
    populate `prompt_tokens` with actual token IDs.
    """
    request_id: str
    prompt_len: int
    output_len: int
    arrival_time: float = 0.0
    prompt_tokens: Optional[List[int]] = None

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len


# ShareGPT-fitted defaults for the synthetic generator. These approximate
# the empirical length distribution used by vLLM's benchmark suite.
_DEFAULT_PROMPT_MU = 5.5
_DEFAULT_PROMPT_SIGMA = 0.8
_DEFAULT_OUTPUT_MU = 4.5
_DEFAULT_OUTPUT_SIGMA = 0.6


def synthetic_workload(
    n_requests: int,
    arrival_rate: float = 1.0,
    prompt_mu: float = _DEFAULT_PROMPT_MU,
    prompt_sigma: float = _DEFAULT_PROMPT_SIGMA,
    output_mu: float = _DEFAULT_OUTPUT_MU,
    output_sigma: float = _DEFAULT_OUTPUT_SIGMA,
    max_prompt_len: int = 2048,
    max_output_len: int = 512,
    seed: int = 0,
) -> Iterator[Request]:
    """Lognormal lengths, Poisson arrivals. No dataset download required.

    Defaults fit ShareGPT. Set `arrival_rate=0` to make all requests
    arrive at t=0 (useful for offline-batch experiments).
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    for i in range(n_requests):
        if arrival_rate > 0:
            t += rng.exponential(1.0 / arrival_rate)
        plen = max(1, min(int(rng.lognormal(prompt_mu, prompt_sigma)), max_prompt_len))
        olen = max(1, min(int(rng.lognormal(output_mu, output_sigma)), max_output_len))
        yield Request(
            request_id=f"syn-{i}",
            prompt_len=plen,
            output_len=olen,
            arrival_time=t,
        )


def fixed_length_workload(
    n_requests: int,
    prompt_len: int,
    output_len: int,
    arrival_rate: float = 0.0,
    seed: int = 0,
) -> Iterator[Request]:
    """All requests have the same prompt and output length."""
    rng = np.random.default_rng(seed)
    t = 0.0
    for i in range(n_requests):
        if arrival_rate > 0:
            t += rng.exponential(1.0 / arrival_rate)
        yield Request(
            request_id=f"fix-{i}",
            prompt_len=prompt_len,
            output_len=output_len,
            arrival_time=t,
        )


def shared_prefix_workload(
    n_parallel: int,
    prompt_len: int,
    output_len: int,
    prompt_tokens: Optional[Sequence[int]] = None,
) -> List[Request]:
    """N requests sharing one identical prompt — for E3 prefix sharing.

    If `prompt_tokens` is provided it's attached to every request so
    content-hashing prefix caches can dedupe. If None, only lengths are
    populated, suitable for allocator-simulation experiments where dedup
    is established by-construction (e.g. via PagedCache.fork_prefix).
    """
    tokens = list(prompt_tokens) if prompt_tokens is not None else None
    if tokens is not None and len(tokens) != prompt_len:
        raise ValueError(
            f"prompt_tokens length {len(tokens)} != prompt_len {prompt_len}"
        )
    return [
        Request(
            request_id=f"shared-{i}",
            prompt_len=prompt_len,
            output_len=output_len,
            arrival_time=0.0,
            prompt_tokens=tokens,
        )
        for i in range(n_parallel)
    ]


def sharegpt_workload(
    json_path: str | Path,
    n_requests: int,
    tokenizer,
    arrival_rate: float = 1.0,
    seed: int = 0,
    min_prompt_len: int = 4,
    max_prompt_len: int = 2048,
) -> Iterator[Request]:
    """Replay real conversations from a ShareGPT JSON dump.

    Get `ShareGPT_V3_unfiltered_cleaned_split.json` from
      huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    (~700MB). For lighter alternatives that stream, see hf_chat_workload.

    Each request uses the first human turn as the prompt and the first
    model turn's length as the target output length.
    """
    rng = np.random.default_rng(seed)
    with open(json_path) as f:
        convs = json.load(f)
    rng.shuffle(convs)

    t = 0.0
    yielded = 0
    for conv in convs:
        if yielded >= n_requests:
            break
        turns = conv.get("conversations", [])
        if len(turns) < 2:
            continue
        human = next((x for x in turns if x.get("from") == "human"), None)
        gpt = next((x for x in turns if x.get("from") == "gpt"), None)
        if human is None or gpt is None:
            continue
        prompt_ids = tokenizer.encode(human["value"])
        if not (min_prompt_len <= len(prompt_ids) <= max_prompt_len):
            continue
        output_ids = tokenizer.encode(gpt["value"])
        if arrival_rate > 0:
            t += rng.exponential(1.0 / arrival_rate)
        yield Request(
            request_id=f"sg-{yielded}",
            prompt_len=len(prompt_ids),
            output_len=max(1, len(output_ids)),
            arrival_time=t,
            prompt_tokens=list(prompt_ids),
        )
        yielded += 1


def hf_chat_workload(
    dataset_name: str,
    n_requests: int,
    tokenizer,
    arrival_rate: float = 1.0,
    seed: int = 0,
    min_prompt_len: int = 4,
    max_prompt_len: int = 2048,
    split: str = "train",
) -> Iterator[Request]:
    """Replay real conversations from any HF dataset that streams.

    Lightweight alternative to sharegpt_workload — no full download required.
    Tested with HuggingFaceH4/no_robots, which has the same prompt/response
    shape as ShareGPT but is small enough to stream a slice from.

    Schemas tried (in order):
      - ex["messages"]: [{"role": ..., "content": ...}, ...]  (e.g. no_robots)
      - ex["conversations"]: [{"from": ..., "value": ...}, ...]  (ShareGPT)
      - ex["instruction"] + ex["response"|"output"]  (Alpaca/Dolly)
    """
    from datasets import load_dataset  # type: ignore

    rng = np.random.default_rng(seed)
    ds = load_dataset(dataset_name, split=split, streaming=True)

    t = 0.0
    yielded = 0
    for ex in ds:
        if yielded >= n_requests:
            break

        prompt_text = response_text = None
        if "messages" in ex:
            msgs = ex["messages"] or []
            user = next((m for m in msgs if m.get("role") == "user"), None)
            asst = next((m for m in msgs if m.get("role") == "assistant"), None)
            if user is not None and asst is not None:
                prompt_text = user.get("content", "")
                response_text = asst.get("content", "")
        elif "conversations" in ex:
            turns = ex["conversations"] or []
            human = next((x for x in turns if x.get("from") == "human"), None)
            gpt = next((x for x in turns if x.get("from") == "gpt"), None)
            if human is not None and gpt is not None:
                prompt_text = human.get("value", "")
                response_text = gpt.get("value", "")
        elif "instruction" in ex:
            prompt_text = ex["instruction"]
            response_text = ex.get("response") or ex.get("output") or ""

        if not prompt_text or response_text is None:
            continue

        prompt_ids = tokenizer.encode(prompt_text)
        if not (min_prompt_len <= len(prompt_ids) <= max_prompt_len):
            continue
        output_ids = tokenizer.encode(response_text)
        if arrival_rate > 0:
            t += rng.exponential(1.0 / arrival_rate)
        yield Request(
            request_id=f"hf-{yielded}",
            prompt_len=len(prompt_ids),
            output_len=max(1, len(output_ids)),
            arrival_time=t,
            prompt_tokens=list(prompt_ids),
        )
        yielded += 1


if __name__ == "__main__":
    # Smoke test: generate a synthetic workload and print its shape.
    reqs = list(synthetic_workload(n_requests=1000, arrival_rate=10.0, seed=42))
    plens = [r.prompt_len for r in reqs]
    olens = [r.output_len for r in reqs]
    print(f"n={len(reqs)}")
    print(f"prompt_len   median={np.median(plens):.0f}  p95={np.percentile(plens, 95):.0f}  max={max(plens)}")
    print(f"output_len   median={np.median(olens):.0f}  p95={np.percentile(olens, 95):.0f}  max={max(olens)}")
    print(f"arrival span: {reqs[-1].arrival_time:.2f}s")

    shared = shared_prefix_workload(n_parallel=8, prompt_len=1024, output_len=128)
    print(f"\nshared-prefix workload: {len(shared)} requests, "
          f"all with prompt_len={shared[0].prompt_len}")
