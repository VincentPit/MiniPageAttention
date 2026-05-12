"""ContinuousScheduler — Phase 2 iteration-level scheduler.

Lifts the v1 uniform-batch limitation: requests join and leave a running
batch independently. One prefill model call admits the next queued request;
one decode model call advances all currently-running sequences. The two
calls happen back-to-back in each scheduler step, with the cache's
`active_indices` flipped between them.

This is the simplest correct continuous-batching design. It is not
chunked-prefill (which would mix prefill + decode in a single forward) —
that's a Phase 2.5 optimization once correctness here is solid.

Run from project root:
  python -m cpu.continuous_scheduler
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from continuous_cache import ContinuousPagedCache  # noqa: E402
from cpu.scheduler import CompletedRequest  # noqa: E402
from bench.workloads import Request  # noqa: E402


@dataclass
class _SequenceState:
    """Internal state for a sequence currently in the running batch."""
    request: Request
    slot: int
    output_tokens: List[int] = field(default_factory=list)
    next_input_token: int = 0
    real_prompt_len: int = 0
    arrival_time: float = 0.0
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0


class ContinuousScheduler:
    def __init__(
        self,
        model,
        tokenizer,
        manager: BlockManager,
        n_slots: int = 8,
        device: str | torch.device = "cpu",
        max_output_len: int = 512,
        sample: bool = False,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        self.cache = ContinuousPagedCache(manager, n_slots=n_slots)
        self.n_slots = n_slots
        self.device = torch.device(device)
        self.max_output_len = max_output_len
        self.sample = sample
        self.temperature = temperature
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
        self.vocab_size = model.config.vocab_size
        self._rng = torch.Generator(device="cpu").manual_seed(seed)

        self.queue: Deque[Request] = deque()
        self.running: List[_SequenceState] = []

    # --- public ----------------------------------------------------------------

    def run(self, requests: List[Request]) -> List[CompletedRequest]:
        for r in requests:
            self.queue.append(r)
        finished_states: List[_SequenceState] = []
        while self.queue or self.running:
            done_in_prefill = self._step_prefill()
            if done_in_prefill is not None:
                finished_states.append(done_in_prefill)
            finished_states.extend(self._step_decode())
        return [self._to_completed(s) for s in finished_states]

    # --- iteration steps -------------------------------------------------------

    def _step_prefill(self) -> Optional[_SequenceState]:
        """Admit one queued request via a B=1 prefill. Returns the new
        sequence's state if it finished immediately (output_len=1 or EOS),
        else None."""
        if not self.queue:
            return None
        free = self.cache.free_slots()
        if not free:
            return None  # batch full, decode will progress and free a slot

        slot = free[0]
        req = self.queue.popleft()
        if req.prompt_tokens is None:
            req.prompt_tokens = torch.randint(
                0, self.vocab_size, (req.prompt_len,), generator=self._rng,
            ).tolist()
        assert req.prompt_tokens is not None

        self.cache.activate(slot)
        self.cache.set_active_indices([slot])

        ids = torch.tensor([req.prompt_tokens], dtype=torch.long, device=self.device)
        L = ids.shape[1]
        attn = torch.ones((1, L), dtype=torch.long, device=self.device)
        pos = torch.arange(L, dtype=torch.long, device=self.device).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=self.cache,
                use_cache=True,
            )
        first = self._next_token(out.logits[:, -1, :])
        t_first = time.time()
        first_tok = int(first.item())

        state = _SequenceState(
            request=req,
            slot=slot,
            output_tokens=[first_tok],
            next_input_token=first_tok,
            real_prompt_len=L,
            arrival_time=req.arrival_time,
            start_time=t0,
            first_token_time=t_first,
        )

        if len(state.output_tokens) >= req.output_len or first_tok == self.eos_id:
            state.end_time = t_first
            self.cache.deactivate(slot)
            return state  # finished without entering decode loop

        self.running.append(state)
        return None

    def _step_decode(self) -> List[_SequenceState]:
        """One decode step over all running sequences. Returns finished states."""
        if not self.running:
            return []
        active = [s.slot for s in self.running]
        self.cache.set_active_indices(active)

        B = len(self.running)
        cache_lens = [self.cache.get_slot_length(s.slot) for s in self.running]
        max_L = max(cache_lens)

        next_tokens = torch.tensor(
            [[s.next_input_token] for s in self.running],
            dtype=torch.long, device=self.device,
        )
        # In the gathered K/V tensor, slot i's new-token K is at column
        # cache_lens[i] (right after its own cache). Columns cache_lens[i]+1
        # ..max_L are zero-pad. So mask=1 for [0, cache_lens[i]] inclusive
        # and 0 for the rest — NOT a global "new token at column max_L".
        # Getting this wrong silently breaks short sequences when batched
        # with longer ones (decode appears to ignore the new query's K).
        attn = torch.zeros((B, max_L + 1), dtype=torch.long, device=self.device)
        for i, l in enumerate(cache_lens):
            attn[i, :l + 1] = 1
        pos = torch.tensor(
            [[l] for l in cache_lens], dtype=torch.long, device=self.device,
        )

        with torch.no_grad():
            out = self.model(
                input_ids=next_tokens,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=self.cache,
                use_cache=True,
            )
        new_tokens = self._next_token(out.logits[:, -1, :])
        t_now = time.time()

        finished: List[_SequenceState] = []
        for i in range(B):
            s = self.running[i]
            tok = int(new_tokens[i].item())
            s.output_tokens.append(tok)
            if (len(s.output_tokens) >= s.request.output_len
                    or tok == self.eos_id):
                s.end_time = t_now
                finished.append(s)
            else:
                s.next_input_token = tok

        for s in finished:
            self.cache.deactivate(s.slot)
            self.running.remove(s)
        return finished

    # --- helpers ---------------------------------------------------------------

    def _next_token(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sample and self.temperature > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs, 1, generator=self._rng).squeeze(-1)
        return logits.argmax(dim=-1)

    def _to_completed(self, s: _SequenceState) -> CompletedRequest:
        return CompletedRequest(
            request_id=s.request.request_id,
            prompt_len=s.request.prompt_len,
            output_tokens=s.output_tokens,
            arrival_time=s.arrival_time,
            start_time=s.start_time,
            first_token_time=s.first_token_time,
            end_time=s.end_time,
        )


# ---------------------------------------------------------------------------
# Smoke test: feed mixed-length prompts, compare each output to model.generate.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("loading gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    cfg = model.config

    mgr = BlockManager(
        n_layers=cfg.n_layer, n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=512, block_size=16,
        dtype=torch.float32, device="cpu",
    )
    sched = ContinuousScheduler(
        model, tokenizer, mgr, n_slots=4, max_output_len=10,
    )

    prompts = [
        "Hello",
        "The capital of France is",
        "Once upon a time",
        "The quick brown fox jumps over the",
    ]

    requests = []
    for i, p in enumerate(prompts):
        ids = tokenizer(p).input_ids
        requests.append(Request(
            request_id=f"r{i}",
            prompt_len=len(ids),
            output_len=8,
            prompt_tokens=ids,
        ))

    completions = sched.run(requests)

    # Reference via HF generate
    n_pass = 0
    by_id = {c.request_id: c for c in completions}
    for i, p in enumerate(prompts):
        ids = tokenizer(p, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        ref = out[0, ids.shape[1]:].tolist()
        c = by_id[f"r{i}"]
        L = min(len(ref), len(c.output_tokens))
        match = ref[:L] == c.output_tokens[:L]
        n_pass += int(match)
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {p!r}  ttft={c.first_token_time - c.start_time:.3f}s")
        if not match:
            print(f"          ref:        {ref}")
            print(f"          continuous: {c.output_tokens}")

    print(f"\n{n_pass}/{len(prompts)} matched")
    print(f"blocks remaining: {mgr.n_used} (expect 0)")
    sys.exit(0 if (n_pass == len(prompts) and mgr.n_used == 0) else 1)
