"""Minimal FastAPI server wrapping ContinuousScheduler.

Synchronous batch endpoint — each HTTP request submits one or more prompts
and waits for all completions. Concurrent HTTP requests are serialized via
a lock; "real" multi-tenant serving (one running scheduler loop, async
requests joining mid-iteration) is a Phase 2.5+ design and not done here.

Run from project root:
  uvicorn serving.api:app --host 0.0.0.0 --port 8000

Smoke test:
  curl -s -X POST http://localhost:8000/v1/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"prompts": ["The capital of France is", "Hello"], "max_tokens": 8}'
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paged_cache import BlockManager  # noqa: E402
from cpu.continuous_scheduler import ContinuousScheduler  # noqa: E402
from bench.workloads import Request  # noqa: E402


# --- request/response schemas ---

class CompletionRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 32


class Completion(BaseModel):
    prompt: str
    text: str
    tokens: List[int]
    output_len: int
    latency_s: float


class CompletionResponse(BaseModel):
    completions: List[Completion]
    device: str


# --- module-level state ---

app = FastAPI(title="PagedAttention demo server",
              description="ContinuousScheduler over a paged KV cache")

_model: Optional[GPT2LMHeadModel] = None
_tokenizer: Optional[GPT2Tokenizer] = None
_scheduler: Optional[ContinuousScheduler] = None
_device: str = "cpu"
_lock = threading.Lock()


def _init():
    """Load model + scheduler. Called on first request."""
    global _model, _tokenizer, _scheduler, _device

    if _scheduler is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    _device = device

    print(f"[api] loading gpt2 on {device} ({dtype})...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=dtype).to(device).eval()
    cfg = model.config
    mgr = BlockManager(
        n_layers=cfg.n_layer, n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=2048, block_size=16,
        dtype=dtype, device=device,
    )
    sched = ContinuousScheduler(
        model, tok, mgr, n_slots=16, device=device,
        max_output_len=512, sample=False,
    )
    _model, _tokenizer, _scheduler = model, tok, sched
    print("[api] ready", flush=True)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "ready": _scheduler is not None, "device": _device}


@app.get("/")
def root():
    return {
        "name": "PagedAttention demo server",
        "endpoints": {
            "POST /v1/completions": "synchronous batch completions",
            "GET /healthz": "readiness check",
        },
    }


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest):
    if not req.prompts:
        raise HTTPException(status_code=400, detail="prompts must be non-empty")
    if req.max_tokens <= 0 or req.max_tokens > 1024:
        raise HTTPException(status_code=400, detail="max_tokens must be in [1, 1024]")

    _init()
    assert _scheduler is not None and _tokenizer is not None

    with _lock:
        requests = []
        for i, p in enumerate(req.prompts):
            ids = _tokenizer(p).input_ids
            if not ids:
                raise HTTPException(status_code=400, detail=f"empty prompt at index {i}")
            requests.append(Request(
                request_id=f"http-{i}",
                prompt_len=len(ids),
                output_len=req.max_tokens,
                prompt_tokens=ids,
            ))
        completions = _scheduler.run(requests)

    by_id = {c.request_id: c for c in completions}
    out: List[Completion] = []
    for i, p in enumerate(req.prompts):
        c = by_id[f"http-{i}"]
        text = _tokenizer.decode(c.output_tokens)
        out.append(Completion(
            prompt=p, text=text, tokens=c.output_tokens,
            output_len=c.output_len, latency_s=c.latency,
        ))
    return CompletionResponse(completions=out, device=_device)
