"""Persistent KV cache store for multi-turn conversation reuse."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx

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

    def save_to_disk(self, cache_dir: Path) -> int:
        """Serialize all cached blocks to disk.

        Writes one safetensors file per block plus a JSON index.
        Returns number of blocks saved.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        index = {"model_name": self.model_name, "block_size": BLOCK_SIZE, "blocks": []}
        saved = 0

        for block_hash, block in self._blocks.items():
            hex_hash = block_hash.hex()
            block_file = cache_dir / f"{hex_hash}.safetensors"

            arrays = {}
            metadata = {}
            for layer_idx, state in enumerate(block.layer_states):
                if state is None or (
                    isinstance(state, tuple)
                    and len(state) == 2
                    and state[0] is None
                    and state[1] is None
                ):
                    metadata[f"layer_{layer_idx}_empty"] = "1"
                    continue
                _serialize_layer_state(arrays, metadata, layer_idx, state)

            if arrays:
                mx.save_safetensors(str(block_file), arrays, metadata)

            index["blocks"].append({
                "hash": hex_hash,
                "token_count": block.token_count,
                "num_layers": len(block.layer_states),
            })
            saved += 1

        with open(cache_dir / "cache_index.json", "w") as f:
            json.dump(index, f, indent=2)

        logger.info("Cache saved: %d blocks to %s", saved, cache_dir)
        return saved

    def load_from_disk(self, cache_dir: Path) -> int:
        """Load cached blocks from disk. Returns number of blocks loaded."""
        index_path = cache_dir / "cache_index.json"
        if not index_path.exists():
            return 0

        with open(index_path) as f:
            index = json.load(f)

        if index.get("model_name") != self.model_name:
            logger.warning(
                "Cache model mismatch: disk=%s, current=%s. Skipping load.",
                index.get("model_name"),
                self.model_name,
            )
            return 0

        loaded = 0
        for block_info in index["blocks"]:
            hex_hash = block_info["hash"]
            block_hash = bytes.fromhex(hex_hash)
            block_file = cache_dir / f"{hex_hash}.safetensors"

            if not block_file.exists():
                continue

            arrays, file_metadata = mx.load(str(block_file), return_metadata=True)
            num_layers = block_info["num_layers"]

            layer_states = []
            for layer_idx in range(num_layers):
                if file_metadata.get(f"layer_{layer_idx}_empty") == "1":
                    layer_states.append((None, None))
                    continue
                state = _deserialize_layer_state(arrays, file_metadata, layer_idx)
                layer_states.append(state)

            self._blocks[block_hash] = _CachedBlock(
                block_hash=block_hash,
                layer_states=layer_states,
                token_count=block_info["token_count"],
            )
            loaded += 1

        logger.info("Cache loaded: %d blocks from %s", loaded, cache_dir)
        return loaded

    def clear(self) -> None:
        self._blocks.clear()

    @property
    def block_count(self) -> int:
        return len(self._blocks)


def _serialize_layer_state(
    arrays: dict, metadata: dict, layer_idx: int, state
) -> None:
    """Serialize a single layer's cache state into arrays dict."""
    if isinstance(state, (list, tuple)) and len(state) >= 2:
        keys, values = state[0], state[1]
        if isinstance(keys, mx.array) and isinstance(values, mx.array):
            # Standard KVCache: (keys, values) tuple of arrays
            metadata[f"layer_{layer_idx}_type"] = "kv"
            arrays[f"layer_{layer_idx}_keys"] = keys
            arrays[f"layer_{layer_idx}_values"] = values
            return
        # Could be NamedTuples (TurboQuant) - serialize each field
        if hasattr(keys, "_fields") and hasattr(values, "_fields"):
            metadata[f"layer_{layer_idx}_type"] = "tq_pair"
            metadata[f"layer_{layer_idx}_keys_type"] = type(keys).__name__
            metadata[f"layer_{layer_idx}_values_type"] = type(values).__name__
            for field_name in keys._fields:
                val = getattr(keys, field_name)
                if isinstance(val, mx.array):
                    arrays[f"layer_{layer_idx}_k_{field_name}"] = val
            for field_name in values._fields:
                val = getattr(values, field_name)
                if isinstance(val, mx.array):
                    arrays[f"layer_{layer_idx}_v_{field_name}"] = val
            return
    # Single NamedTuple state
    if hasattr(state, "_fields"):
        metadata[f"layer_{layer_idx}_type"] = type(state).__name__
        for field_name in state._fields:
            val = getattr(state, field_name)
            if isinstance(val, mx.array):
                arrays[f"layer_{layer_idx}_{field_name}"] = val


def _deserialize_layer_state(arrays: dict, metadata: dict, layer_idx: int):
    """Deserialize a single layer's cache state from arrays dict."""
    state_type = metadata.get(f"layer_{layer_idx}_type", "kv")

    if state_type == "kv":
        keys = arrays.get(f"layer_{layer_idx}_keys")
        values = arrays.get(f"layer_{layer_idx}_values")
        return (keys, values)

    if state_type == "tq_pair":
        k_prefix = f"layer_{layer_idx}_k_"
        v_prefix = f"layer_{layer_idx}_v_"
        k_arrays = {
            k[len(k_prefix):]: v
            for k, v in arrays.items()
            if k.startswith(k_prefix)
        }
        v_arrays = {
            k[len(v_prefix):]: v
            for k, v in arrays.items()
            if k.startswith(v_prefix)
        }

        keys_type_name = metadata.get(f"layer_{layer_idx}_keys_type")
        values_type_name = metadata.get(f"layer_{layer_idx}_values_type")
        keys_state = _rebuild_namedtuple(keys_type_name, k_arrays)
        values_state = _rebuild_namedtuple(values_type_name, v_arrays)
        return (keys_state, values_state)

    # Unknown type, try to reconstruct as plain tuple
    keys = arrays.get(f"layer_{layer_idx}_keys")
    values = arrays.get(f"layer_{layer_idx}_values")
    if keys is not None and values is not None:
        return (keys, values)
    return None


# Registry for deserializing NamedTuples by name.
_NAMEDTUPLE_REGISTRY: dict[str, type] = {}


def _ensure_namedtuple_registry():
    if _NAMEDTUPLE_REGISTRY:
        return
    try:
        from mlx_vlm.turboquant import (
            TurboQuantMSEState,
            TurboQuantProdState,
        )
        _NAMEDTUPLE_REGISTRY["TurboQuantMSEState"] = TurboQuantMSEState
        _NAMEDTUPLE_REGISTRY["TurboQuantProdState"] = TurboQuantProdState
    except ImportError:
        pass


def _rebuild_namedtuple(type_name: str, field_arrays: dict):
    _ensure_namedtuple_registry()
    cls = _NAMEDTUPLE_REGISTRY.get(type_name)
    if cls is None:
        return tuple(field_arrays.values())
    return cls(**field_arrays)
