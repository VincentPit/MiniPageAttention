"""ContinuousPagedCache: variable-length, dynamic-membership cache for Phase 2.

Differences from the v1 PagedCache:
  - Slots are addressed by index, with explicit activate/deactivate.
  - At any moment a subset of slots is "active" — the active set is what
    the next model() call operates on. Set via set_active_indices().
  - Active slots can have *different* cache lengths. _gather returns a
    [B, H, max_L, D] tensor zero-padded for shorter slots; the scheduler
    must propagate a per-row attention_mask so attention ignores the pad.

The same BlockManager is reused — ContinuousPagedCache and PagedCache can
even share a pool if needed. Refcounting and fork_prefix work the same way.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from paged_cache import BlockManager


class ContinuousPagedLayer(CacheLayerMixin):
    """One transformer layer's slice of a ContinuousPagedCache.

    Holds per-slot block tables and lengths. At any time a subset
    `active_indices` is selected; update() and _gather() operate on those.

    Inactive slot: tables[s] is None and lens[s] == -1.
    Active but empty (just activated): tables[s] = [], lens[s] = 0.
    """

    is_sliding = False
    layer_type = "paged-continuous"
    is_compileable = False

    def __init__(self, manager: BlockManager, layer_idx: int, n_slots: int):
        super().__init__()
        self.manager = manager
        self.layer_idx = layer_idx
        self.n_slots = n_slots
        self.tables: List[Optional[List[int]]] = [None] * n_slots
        self.lens: List[int] = [-1] * n_slots
        self.active_indices: List[int] = []
        self.is_initialized = True
        self.dtype = manager.k_pool.dtype
        self.device = manager.k_pool.device

    # --- HF Cache layer API -------------------------------------------------

    def lazy_initialization(self, key_states, value_states) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,    # [B, H, T_new, D] — B == len(active_indices)
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T_new, D = key_states.shape
        assert B == len(self.active_indices), (
            f"update batch {B} != active_indices {len(self.active_indices)}"
        )
        bs = self.manager.block_size

        for i, slot in enumerate(self.active_indices):
            assert self.lens[slot] >= 0, f"slot {slot} not active"
            for t in range(T_new):
                pos = self.lens[slot] + t
                slot_in_block = pos % bs
                if slot_in_block == 0:
                    table = self.tables[slot]
                    assert table is not None
                    table.append(self.manager.allocate())
                table = self.tables[slot]
                assert table is not None
                bid = table[-1]
                self.manager.k_pool[self.layer_idx, bid, slot_in_block] = key_states[i, :, t]
                self.manager.v_pool[self.layer_idx, bid, slot_in_block] = value_states[i, :, t]
            self.lens[slot] += T_new

        return self._gather()

    def _gather(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return [B, H, max_L, D] for active slots, zero-padded to max_L."""
        H = self.manager.n_heads
        D = self.manager.head_dim

        active_lens = [self.lens[s] for s in self.active_indices]
        max_L = max(active_lens) if active_lens else 0
        if max_L == 0:
            empty = torch.zeros(
                (len(self.active_indices), H, 0, D),
                dtype=self.dtype, device=self.device,
            )
            return empty, empty

        Ks, Vs = [], []
        for slot in self.active_indices:
            L = self.lens[slot]
            bids = self.tables[slot]
            if bids:
                k = self.manager.k_pool[self.layer_idx, bids].reshape(-1, H, D)[:L]
                v = self.manager.v_pool[self.layer_idx, bids].reshape(-1, H, D)[:L]
            else:
                k = torch.zeros((0, H, D), dtype=self.dtype, device=self.device)
                v = torch.zeros((0, H, D), dtype=self.dtype, device=self.device)
            if L < max_L:
                pad = torch.zeros(
                    (max_L - L, H, D), dtype=self.dtype, device=self.device,
                )
                k = torch.cat([k, pad], dim=0)
                v = torch.cat([v, pad], dim=0)
            Ks.append(k)
            Vs.append(v)
        # [B, max_L, H, D] -> [B, H, max_L, D]
        K = torch.stack(Ks).permute(0, 2, 1, 3).contiguous()
        V = torch.stack(Vs).permute(0, 2, 1, 3).contiguous()
        return K, V

    def append_only_decode(
        self,
        key_states: torch.Tensor,    # [B_active, H, 1, D]
        value_states: torch.Tensor,  # [B_active, H, 1, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Phase 3 path: write new K/V into the paged pool, then return
        # block_tables and seq_lens (over active slots) for the kernel.
        B, H, T, D = key_states.shape
        assert T == 1, "append_only_decode is decode-step only (T_new=1)"
        assert B == len(self.active_indices)
        bs = self.manager.block_size

        for i, slot in enumerate(self.active_indices):
            assert self.lens[slot] >= 0, f"slot {slot} not active"
            pos = self.lens[slot]
            in_blk = pos % bs
            if in_blk == 0:
                table = self.tables[slot]
                assert table is not None
                table.append(self.manager.allocate())
            table = self.tables[slot]
            assert table is not None
            bid = table[-1]
            self.manager.k_pool[self.layer_idx, bid, in_blk] = key_states[i, :, 0]
            self.manager.v_pool[self.layer_idx, bid, in_blk] = value_states[i, :, 0]
            self.lens[slot] += 1

        return self.build_metadata_active()

    def build_metadata_active(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(block_tables, seq_lens) over active_indices, for the kernel."""
        active_lens = [self.lens[s] for s in self.active_indices]
        active_tables = [self.tables[s] or [] for s in self.active_indices]
        max_blocks = max((len(t) for t in active_tables), default=0)
        device = self.manager.k_pool.device
        block_tables = torch.full(
            (len(active_tables), max_blocks), -1, dtype=torch.int32, device=device,
        )
        for i, tbl in enumerate(active_tables):
            if tbl:
                block_tables[i, :len(tbl)] = torch.tensor(
                    tbl, dtype=torch.int32, device=device,
                )
        seq_lens = torch.tensor(active_lens, dtype=torch.int32, device=device)
        return block_tables, seq_lens

    def get_seq_length(self) -> int:
        if not self.active_indices:
            return 0
        return max(self.lens[s] for s in self.active_indices)

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def reorder_cache(self, beam_idx) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        for slot in range(self.n_slots):
            self._free_slot(slot)
        self.active_indices = []

    # --- continuous-only ----------------------------------------------------

    def activate(self, slot: int) -> None:
        assert self.lens[slot] == -1, f"slot {slot} already active"
        self.tables[slot] = []
        self.lens[slot] = 0

    def deactivate(self, slot: int) -> None:
        self._free_slot(slot)

    def _free_slot(self, slot: int) -> None:
        table = self.tables[slot]
        if table is not None:
            for bid in table:
                self.manager.free(bid)
        self.tables[slot] = None
        self.lens[slot] = -1


class ContinuousPagedCache(Cache):
    """Variable-length, dynamic-membership PagedCache.

    Workflow:
      1. cache.activate(slot)           # claim a slot
      2. cache.set_active_indices([s])  # select for the next model call
      3. model(..., past_key_values=cache, use_cache=True)
      4. ... repeat with different active sets ...
      5. cache.deactivate(slot)         # frees blocks back to BlockManager
    """

    def __init__(self, manager: BlockManager, n_slots: int):
        layers = [
            ContinuousPagedLayer(manager, i, n_slots)
            for i in range(manager.n_layers)
        ]
        super().__init__(layers=layers)
        self.manager = manager
        self.n_slots = n_slots

    def set_active_indices(self, indices: List[int]) -> None:
        for layer in self.layers:
            layer.active_indices = list(indices)

    def activate(self, slot: int) -> None:
        for layer in self.layers:
            layer.activate(slot)

    def deactivate(self, slot: int) -> None:
        for layer in self.layers:
            layer.deactivate(slot)

    def get_slot_length(self, slot: int) -> int:
        """Per-slot length (vs get_seq_length which returns max over active)."""
        return self.layers[0].lens[slot]

    def is_active(self, slot: int) -> bool:
        return self.layers[0].lens[slot] >= 0

    def free_slots(self) -> List[int]:
        return [s for s in range(self.n_slots) if not self.is_active(s)]
