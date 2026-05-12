"""Monkey-patch GPT-2's self-attention to route decode steps through the
paged-attention kernel.

When the patch is enabled and the forward sees:
  - hidden_states.shape[1] == 1   (decode step)
  - past_key_values is a paged cache (PagedCache or ContinuousPagedCache,
    duck-typed via .layers[0].append_only_decode)
  - not cross-attention, not output_attentions

it skips HF's `cache.update()` + `eager_attention_forward` path and
instead:
  1. projects Q/K/V from c_attn,
  2. appends the new K/V into the paged pool via append_only_decode,
  3. calls paged_attention_decode(q, k_pool[layer], v_pool[layer],
     block_tables, seq_lens, scale).

Prefill (T>1) and any non-paged cache fall through to the unmodified HF
forward, which still goes through PagedCache.update()._gather() — that's
fine, prefill is one-shot per request and the gather cost is amortized.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from gpu.paged_attention_kernel import paged_attention_decode

_ORIG_ATTR = "_paged_attn_original_forward"


def _is_paged_cache(cache) -> bool:
    if cache is None:
        return False
    layers = getattr(cache, "layers", None)
    if not layers:
        return False
    return hasattr(layers[0], "append_only_decode")


def _paged_decode_forward(
    self: GPT2Attention,
    hidden_states,
    past_key_values=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    take_paged = (
        encoder_hidden_states is None
        and not output_attentions
        and past_key_values is not None
        and _is_paged_cache(past_key_values)
        and hidden_states is not None
        and hidden_states.shape[1] == 1
    )
    if not take_paged:
        return getattr(GPT2Attention, _ORIG_ATTR)(
            self, hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )

    B, T, _ = hidden_states.shape  # T == 1
    H = self.num_heads
    D = self.head_dim

    qkv = self.c_attn(hidden_states)
    q, k, v = qkv.split(self.split_size, dim=2)
    q = q.view(B, T, H, D).transpose(1, 2).contiguous()  # [B, H, 1, D]
    k = k.view(B, T, H, D).transpose(1, 2).contiguous()
    v = v.view(B, T, H, D).transpose(1, 2).contiguous()

    layer = past_key_values.layers[self.layer_idx]
    block_tables, seq_lens = layer.append_only_decode(k, v)

    manager = past_key_values.manager
    key_cache = manager.k_pool[self.layer_idx]    # [n_blocks, bs, H, D]
    value_cache = manager.v_pool[self.layer_idx]

    q_kernel = q.squeeze(2)                       # [B, H, D]
    out = paged_attention_decode(
        q_kernel, key_cache, value_cache,
        block_tables, seq_lens, float(self.scaling),
    )

    # [B, H, D] -> [B, 1, H, D] -> [B, 1, H*D]
    attn_output = out.unsqueeze(1).reshape(B, T, H * D)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    return attn_output, None


def enable_paged_attention() -> None:
    """Idempotent."""
    if hasattr(GPT2Attention, _ORIG_ATTR):
        return
    setattr(GPT2Attention, _ORIG_ATTR, GPT2Attention.forward)
    GPT2Attention.forward = _paged_decode_forward  # type: ignore[assignment]


def disable_paged_attention() -> None:
    if not hasattr(GPT2Attention, _ORIG_ATTR):
        return
    GPT2Attention.forward = getattr(GPT2Attention, _ORIG_ATTR)  # type: ignore[assignment]
    delattr(GPT2Attention, _ORIG_ATTR)


class paged_attention:
    """Context manager to scope the patch."""
    def __enter__(self):
        enable_paged_attention()
        return self

    def __exit__(self, *exc):
        disable_paged_attention()
        return False
