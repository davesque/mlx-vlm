"""Persistent KV cache store for multi-turn conversation reuse."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

BLOCK_SIZE = 1024

logger = logging.getLogger(__name__)


def compute_block_hash(
    parent_hash: Optional[bytes],
    token_ids: List[int],
    model_name: Optional[str] = None,
) -> bytes:
    """Compute a content-based hash for a block of tokens.

    Each block's hash encodes its full prefix context via the parent
    hash chain, enabling O(1) prefix matching.
    """
    hasher = hashlib.sha256()
    if model_name:
        hasher.update(model_name.encode("utf-8"))
    if parent_hash:
        hasher.update(parent_hash)
    else:
        hasher.update(b"mlx-vlm-root")
    hasher.update(bytes(str(tuple(token_ids)), "utf-8"))
    return hasher.digest()


@dataclass
class _CachedBlock:
    """One block's worth of KV cache data, keyed by block hash."""

    block_hash: bytes
    # Per-layer list of cache.state snapshots at this block boundary.
    layer_states: list
    token_count: int
    last_access: float = field(default_factory=time.time)


class PromptCacheStore:
    """In-memory KV cache store with block-level prefix matching.

    Stores KV cache state at block boundaries (every BLOCK_SIZE tokens).
    On lookup, finds the longest matching prefix by walking the block
    hash chain. Provides LRU eviction by block count.
    """

    def __init__(
        self,
        model_name: str = "",
        max_entries: int = 256,
    ):
        self.model_name = model_name
        self.max_entries = max_entries
        self._blocks: OrderedDict[bytes, _CachedBlock] = OrderedDict()

    def put(
        self,
        token_ids: list[int],
        prompt_cache: list[Any],
    ) -> int:
        """Store cache state, returning number of blocks stored."""
        num_full_blocks = len(token_ids) // BLOCK_SIZE
        if num_full_blocks == 0:
            return 0

        stored = 0
        parent_hash = None

        for i in range(num_full_blocks):
            start = i * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block_tokens = token_ids[start:end]
            block_hash = compute_block_hash(
                parent_hash, block_tokens, model_name=self.model_name,
            )

            if block_hash not in self._blocks:
                # Snapshot each layer's state at this point
                layer_states = []
                for cache_obj in prompt_cache:
                    state = cache_obj.state
                    layer_states.append(state)

                self._blocks[block_hash] = _CachedBlock(
                    block_hash=block_hash,
                    layer_states=layer_states,
                    token_count=end,
                )
                self._evict_if_needed()
                stored += 1
            else:
                self._blocks.move_to_end(block_hash)
                self._blocks[block_hash].last_access = time.time()

            parent_hash = block_hash

        if stored > 0:
            logger.info(
                "Cache store: saved %d new blocks (%d tokens, total blocks=%d)",
                stored,
                num_full_blocks * BLOCK_SIZE,
                len(self._blocks),
            )
        return stored

    def get(
        self,
        token_ids: list[int],
    ) -> Optional[Tuple[list[Any], int]]:
        """Find the longest cached prefix for the given tokens.

        Returns (layer_states, num_matched_tokens) or None if no match.
        """
        num_full_blocks = len(token_ids) // BLOCK_SIZE
        if num_full_blocks == 0:
            return None

        parent_hash = None
        last_match: Optional[tuple[_CachedBlock, int]] = None

        for i in range(num_full_blocks):
            start = i * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block_tokens = token_ids[start:end]
            block_hash = compute_block_hash(
                parent_hash, block_tokens, model_name=self.model_name,
            )

            if block_hash not in self._blocks:
                break

            self._blocks.move_to_end(block_hash)
            self._blocks[block_hash].last_access = time.time()
            last_match = (self._blocks[block_hash], end)
            parent_hash = block_hash

        if last_match is None:
            return None

        block, num_matched_tokens = last_match
        logger.info(
            "Cache hit: %d tokens (%d blocks)",
            num_matched_tokens,
            num_matched_tokens // BLOCK_SIZE,
        )
        return block.layer_states, num_matched_tokens

    def reconstruct_cache(
        self,
        layer_states: list,
        cache_template: Optional[list[Any]] = None,
    ) -> list[Any]:
        """Reconstruct a prompt_cache list from stored layer states.

        Args:
            layer_states: Per-layer state snapshots from get().
            cache_template: Optional list of empty cache objects to populate.
                If None, creates KVCache objects for each layer.

        Returns:
            List of cache objects with state restored, ready for generate_step.
        """
        from mlx_lm.models.cache import KVCache

        result = []
        for i, state in enumerate(layer_states):
            if cache_template and i < len(cache_template):
                cache_obj = cache_template[i]
            else:
                cache_obj = KVCache()
            cache_obj.state = state
            result.append(cache_obj)
        return result

    def _evict_if_needed(self) -> None:
        while len(self._blocks) > self.max_entries:
            evicted_hash, evicted = self._blocks.popitem(last=False)
            logger.debug("Cache eviction: block %s", evicted_hash.hex()[:16])

    def clear(self) -> None:
        self._blocks.clear()

    @property
    def block_count(self) -> int:
        return len(self._blocks)
