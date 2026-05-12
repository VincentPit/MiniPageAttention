"""
PagedCache: a HuggingFace `Cache` backed by a paged KV pool.

v1 scope (deliberately small):
  - Same-length-batch only (every sequence in a batch advances together).
  - Decode + prefill both go through the same path; no custom kernel.
  - K/V are gathered into contiguous [B, H, T, D] before being returned to
    HF's stock attention. Preserves the *allocation* behaviour of
    PagedAttention without writing a custom kernel.

Compatible with transformers >= 5.0 (Cache-as-container-of-layers API).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class BlockManager:
    """Global pool of fixed-size KV blocks shared across layers and sequences."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        n_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_blocks = n_blocks
        self.block_size = block_size

        shape = (n_layers, n_blocks, block_size, n_heads, head_dim)
        self.k_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=device)

        self._free: List[int] = list(range(n_blocks))
        self._refs: List[int] = [0] * n_blocks

        # Optional hooks. Each is called as f(block_id, n_used_after_op).
        # Used by E2 fragmentation experiments to record allocation traces.
        self.on_alloc = None
        self.on_free = None

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError(f"Block pool exhausted ({self.n_blocks} blocks)")
        bid = self._free.pop()
        self._refs[bid] = 1
        if self.on_alloc is not None:
            self.on_alloc(bid, self.n_used)
        return bid

    def fork(self, bid: int) -> int:
        self._refs[bid] += 1
        return bid

    def free(self, bid: int) -> None:
        self._refs[bid] -= 1
        if self._refs[bid] == 0:
            self._free.append(bid)
            if self.on_free is not None:
                self.on_free(bid, self.n_used)

    @property
    def n_used(self) -> int:
        return self.n_blocks - len(self._free)


class PagedLayer(CacheLayerMixin):
    """One transformer layer's slice of a PagedCache.

    Holds the per-sequence block tables and lengths for this layer; delegates
    physical storage to a shared BlockManager indexed by `layer_idx`.
    """

    is_sliding = False
    layer_type = "paged"
    is_compileable = False

    def __init__(self, manager: BlockManager, layer_idx: int, batch_size: int):
        super().__init__()
        self.manager = manager
        self.layer_idx = layer_idx
        self.batch_size = batch_size
        self.tables: List[List[int]] = [[] for _ in range(batch_size)]
        self.lens: List[int] = [0] * batch_size
        self.is_initialized = True
        self.dtype = manager.k_pool.dtype
        self.device = manager.k_pool.device

    def lazy_initialization(self, key_states, value_states) -> None:
        # Pools are already allocated; just record the dtype/device the
        # model is using in case it differs from what BlockManager was sized for.
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,    # [B, H, T_new, D]
        value_states: torch.Tensor,  # [B, H, T_new, D]
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T_new, D = key_states.shape
        assert B == self.batch_size, f"batch size mismatch: {B} vs {self.batch_size}"
        bs = self.manager.block_size

        for b in range(B):
            for t in range(T_new):
                pos = self.lens[b] + t
                slot = pos % bs
                if slot == 0:
                    self.tables[b].append(self.manager.allocate())
                bid = self.tables[b][-1]
                self.manager.k_pool[self.layer_idx, bid, slot] = key_states[b, :, t]
                self.manager.v_pool[self.layer_idx, bid, slot] = value_states[b, :, t]
            self.lens[b] += T_new

        return self._gather()

    def _gather(self) -> Tuple[torch.Tensor, torch.Tensor]:
        L = self.lens[0]  # v1: uniform batch
        Ks, Vs = [], []
        for b in range(self.batch_size):
            bids = self.tables[b]
            k = self.manager.k_pool[self.layer_idx, bids].reshape(
                -1, self.manager.n_heads, self.manager.head_dim,
            )
            v = self.manager.v_pool[self.layer_idx, bids].reshape(
                -1, self.manager.n_heads, self.manager.head_dim,
            )
            Ks.append(k[:L])
            Vs.append(v[:L])
        # [B, L, H, D] -> [B, H, L, D]
        K = torch.stack(Ks).permute(0, 2, 1, 3).contiguous()
        V = torch.stack(Vs).permute(0, 2, 1, 3).contiguous()
        return K, V

    def append_only_decode(
        self,
        key_states: torch.Tensor,    # [B, H, 1, D]
        value_states: torch.Tensor,  # [B, H, 1, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Phase 3 path: write the new K/V into the paged pool, then return
        # block_tables and seq_lens for the kernel — no _gather().
        B, H, T, D = key_states.shape
        assert T == 1, "append_only_decode is decode-step only (T_new=1)"
        assert B == self.batch_size
        bs = self.manager.block_size

        for b in range(B):
            pos = self.lens[b]
            slot = pos % bs
            if slot == 0:
                self.tables[b].append(self.manager.allocate())
            bid = self.tables[b][-1]
            self.manager.k_pool[self.layer_idx, bid, slot] = key_states[b, :, 0]
            self.manager.v_pool[self.layer_idx, bid, slot] = value_states[b, :, 0]
            self.lens[b] += 1

        return self.build_metadata(list(range(B)))

    def build_metadata(self, batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (block_tables, seq_lens) for the kernel.

        block_tables: int32 [B, max_blocks_per_seq], -1 padded.
        seq_lens:     int32 [B].
        """
        tables = [self.tables[b] for b in batch_indices]
        lens = [self.lens[b] for b in batch_indices]
        max_blocks = max((len(t) for t in tables), default=0)
        device = self.manager.k_pool.device
        block_tables = torch.full(
            (len(tables), max_blocks), -1, dtype=torch.int32, device=device,
        )
        for i, tbl in enumerate(tables):
            if tbl:
                block_tables[i, :len(tbl)] = torch.tensor(
                    tbl, dtype=torch.int32, device=device,
                )
        seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
        return block_tables, seq_lens

    def get_seq_length(self) -> int:
        return self.lens[0]

    def get_max_cache_shape(self) -> int:
        return -1  # dynamic

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        raise NotImplementedError("reorder_cache not implemented in PagedLayer v1")

    def reset(self) -> None:
        for b in range(self.batch_size):
            for bid in self.tables[b]:
                self.manager.free(bid)
            self.tables[b] = []
            self.lens[b] = 0


class PagedCache(Cache):
    def __init__(self, manager: BlockManager, batch_size: int):
        layers = [PagedLayer(manager, i, batch_size) for i in range(manager.n_layers)]
        super().__init__(layers=layers)
        self.manager = manager
        self.batch_size = batch_size

    def fork_prefix(
        self,
        src_seq: int,
        dst_seq: int,
        n_tokens: int,
        src_cache: Optional["PagedCache"] = None,
    ) -> None:
        """Share the first n_tokens of a source sequence's KV with dst_seq.

        If `src_cache` is None, `src_seq` refers to this cache. Pass `src_cache`
        to fork from a *different* cache (e.g., a B=1 prefill cache into a B=N
        decode cache for parallel sampling). Both caches must share the same
        BlockManager.

        Only whole blocks can be shared safely; n_tokens is rounded down.
        """
        src_cache = src_cache if src_cache is not None else self
        if self.manager is not src_cache.manager:
            raise ValueError("fork_prefix requires both caches share a BlockManager")
        bs = self.manager.block_size
        n_full = n_tokens // bs
        for self_layer, src_layer in zip(self.layers, src_cache.layers):
            src_table = src_layer.tables[src_seq]
            dst_table = self_layer.tables[dst_seq]
            if dst_table:
                raise ValueError(f"dst_seq {dst_seq} must start empty (has {len(dst_table)} blocks)")
            for i in range(n_full):
                self.manager.fork(src_table[i])
                dst_table.append(src_table[i])
            self_layer.lens[dst_seq] = n_full * bs

    def free_sequence(self, seq_idx: int) -> None:
        for layer in self.layers:
            for bid in layer.tables[seq_idx]:
                self.manager.free(bid)
            layer.tables[seq_idx] = []
            layer.lens[seq_idx] = 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = (
        GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=dtype).to(device).eval()
    )
    cfg = model.config

    mgr = BlockManager(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        n_blocks=512,
        block_size=16,
        dtype=dtype,
        device=device,
    )
    cache = PagedCache(mgr, batch_size=1)

    prompt = tok("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**prompt, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        print("first token:", tok.decode(next_id[0]))
        print("seq_len after prefill:", cache.layers[0].get_seq_length())
        print("blocks used:", mgr.n_used, "/", mgr.n_blocks)

        out2 = model(input_ids=next_id, past_key_values=cache, use_cache=True)
        print("seq_len after 1 decode:", cache.layers[0].get_seq_length())
        print("blocks used:", mgr.n_used, "/", mgr.n_blocks)
