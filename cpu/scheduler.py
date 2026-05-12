"""CPU scheduler for Project A Phase 1.

Runs a list of Requests through (HF model + PagedCache) as a single
uniform-decoding batch. Records per-request metrics (TTFT, latency,
output tokens) for E1, E3, and E4.

Limitations inherited from PagedCache v1 (uniform-batch only):
  - All requests in a run_batch() advance in lockstep. A sequence that
    finishes early continues to occupy a cache slot until the longest
    sequence finishes; its post-finish tokens are discarded.
  - Variable-length prompts are handled by left-padding with EOS plus
    an attention_mask, so the cache sees a uniform length.

Phase 2 will lift these by adding continuous batching at the cache layer.

Run from project root:
  python -m cpu.scheduler
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from paged_cache import BlockManager, PagedCache
from bench.workloads import Request


@dataclass
class CompletedRequest:
    request_id: str
    prompt_len: int
    output_tokens: List[int]
    arrival_time: float
    start_time: float        # when scheduler picked this batch up
    first_token_time: float  # absolute time of first generated token
    end_time: float          # absolute time of last generated token

    @property
    def ttft(self) -> float:
        """Time to first token, measured from when the scheduler started this batch.

        Phase 1 is offline batch — everyone is "available" at start_time, so we
        ignore Request.arrival_time here. Online replay (Phase 2) will need a
        separate queue-time metric that respects arrivals.
        """
        return self.first_token_time - self.start_time

    @property
    def latency(self) -> float:
        """End-to-end latency from scheduler start (see ttft note)."""
        return self.end_time - self.start_time

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)


class Scheduler:
    def __init__(
        self,
        model,
        tokenizer,
        manager: BlockManager,
        device: str | torch.device = "cpu",
        max_output_len: int = 512,
        sample: bool = False,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        self.device = torch.device(device)
        self.max_output_len = max_output_len
        self.sample = sample
        self.temperature = temperature
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_token_id
        self.vocab_size = model.config.vocab_size
        self._rng = torch.Generator(device="cpu").manual_seed(seed)

    def run_batch(self, requests: List[Request]) -> List[CompletedRequest]:
        """Run a single uniform-decoding batch. One CompletedRequest per input."""
        if not requests:
            return []
        B = len(requests)

        # Fill in token IDs for synthetic requests.
        for r in requests:
            if r.prompt_tokens is None:
                r.prompt_tokens = torch.randint(
                    0, self.vocab_size, (r.prompt_len,), generator=self._rng,
                ).tolist()
            assert r.prompt_tokens is not None  # narrow for the type checker

        # Left-pad prompts to a common length; build attention mask.
        real_lens = [len(r.prompt_tokens) for r in requests]
        max_prompt = max(real_lens)
        input_ids = torch.full(
            (B, max_prompt), self.pad_token_id, dtype=torch.long, device=self.device,
        )
        attn_mask = torch.zeros((B, max_prompt), dtype=torch.long, device=self.device)
        for i, r in enumerate(requests):
            L = real_lens[i]
            input_ids[i, max_prompt - L:] = torch.tensor(
                r.prompt_tokens, dtype=torch.long, device=self.device,
            )
            attn_mask[i, max_prompt - L:] = 1

        # GPT-2 uses learned position embeddings, so position_ids must reflect
        # *real* positions, not slot indices. For left-padded inputs:
        # cumsum(mask)-1 gives [0..0, 0, 1, 2, ..., L-1] for each row; clamp
        # at 0 to keep padded positions valid (they're masked out anyway).
        position_ids = attn_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp(min=0)
        real_lens_tensor = torch.tensor(real_lens, dtype=torch.long, device=self.device)

        cache = PagedCache(self.manager, batch_size=B)

        target_lens = [r.output_len for r in requests]
        outputs: List[List[int]] = [[] for _ in range(B)]
        first_token_t: List[Optional[float]] = [None] * B
        end_t: List[Optional[float]] = [None] * B
        done = [False] * B

        try:
            start_t = time.time()

            # --- Prefill ---
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            next_tokens = self._next_token(out.logits[:, -1, :])
            t_first = time.time()

            for i in range(B):
                outputs[i].append(int(next_tokens[i].item()))
                first_token_t[i] = t_first
                if (len(outputs[i]) >= target_lens[i]
                        or next_tokens[i].item() == self.eos_token_id):
                    done[i] = True
                    end_t[i] = t_first

            # --- Decode loop ---
            for step in range(1, self.max_output_len):
                if all(done):
                    break
                # Extend mask for the new token (real for everyone, including
                # finished sequences whose extra outputs we'll discard).
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((B, 1), dtype=torch.long, device=self.device)],
                    dim=1,
                )
                # Decode-step input is the token at real position
                # real_lens[i] + (step - 1) for sequence i.
                decode_position_ids = (real_lens_tensor + (step - 1)).unsqueeze(-1)
                with torch.no_grad():
                    out = self.model(
                        input_ids=next_tokens.unsqueeze(-1),  # [B, 1]
                        attention_mask=attn_mask,
                        position_ids=decode_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                next_tokens = self._next_token(out.logits[:, -1, :])
                t_now = time.time()

                for i in range(B):
                    if done[i]:
                        continue
                    tok = int(next_tokens[i].item())
                    outputs[i].append(tok)
                    if (len(outputs[i]) >= target_lens[i] or tok == self.eos_token_id):
                        done[i] = True
                        end_t[i] = t_now

            t_final = time.time()
            for i in range(B):
                if end_t[i] is None:
                    end_t[i] = t_final
        finally:
            for i in range(B):
                cache.free_sequence(i)

        return [
            CompletedRequest(
                request_id=requests[i].request_id,
                prompt_len=requests[i].prompt_len,
                output_tokens=outputs[i],
                arrival_time=requests[i].arrival_time,
                start_time=start_t,
                first_token_time=first_token_t[i],
                end_time=end_t[i],
            )
            for i in range(B)
        ]

    def _next_token(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sample and self.temperature > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(
                probs, num_samples=1, generator=self._rng,
            ).squeeze(-1)
        return logits.argmax(dim=-1)


def aggregate_stats(completions: List[CompletedRequest]) -> dict:
    """Aggregate per-request metrics into a single summary."""
    from bench.metrics import LatencyStats

    if not completions:
        return {"n": 0}
    total_out = sum(c.output_len for c in completions)
    wallclock = max(c.end_time for c in completions) - min(c.start_time for c in completions)
    return {
        "n": len(completions),
        "total_output_tokens": total_out,
        "tokens_per_second": total_out / wallclock if wallclock > 0 else float("inf"),
        "ttft": LatencyStats.from_latencies(c.ttft for c in completions),
        "latency": LatencyStats.from_latencies(c.latency for c in completions),
    }


# ---------------------------------------------------------------------------
# Smoke test: load GPT-2, run a small batch end-to-end.
# Run from project root: python -m cpu.scheduler
# Requires: pip install torch transformers numpy
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from bench.workloads import fixed_length_workload

    print("loading gpt2...")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    cfg = model.config

    mgr = BlockManager(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=128,
        block_size=16,
        dtype=torch.float32,
        device="cpu",
    )

    sched = Scheduler(model, tok, mgr, device="cpu", max_output_len=12)

    reqs = list(fixed_length_workload(n_requests=2, prompt_len=8, output_len=10))
    completions = sched.run_batch(reqs)

    print(f"\nblocks used after run: {mgr.n_used} / {mgr.n_blocks}  (should be 0)")
    print(f"aggregate: {aggregate_stats(completions)}\n")
    for c in completions:
        text = tok.decode(c.output_tokens)
        print(f"  {c.request_id}: ttft={c.ttft:.3f}s lat={c.latency:.3f}s "
              f"out_len={c.output_len}  text={text!r}")
