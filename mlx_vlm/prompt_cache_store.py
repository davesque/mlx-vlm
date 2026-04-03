"""Persistent KV cache store for multi-turn conversation reuse.

Stores full KV cache snapshots keyed by token sequence. Supports
multiple concurrent conversations with prefix-based matching (e.g.
forked sessions share their common prefix cache).

Design: no fixed block size. Each entry stores the exact token sequence
and the full cache state. On lookup, finds the stored entry whose tokens
are the longest prefix of the new request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """A cached KV state for a specific token sequence."""

    token_ids: list[int]
    layer_states: list  # per-layer cache.state snapshots
    last_access: float = field(default_factory=time.time)


class PromptCacheStore:
    """In-memory KV cache store with prefix matching.

    Stores KV cache state for complete token sequences. On lookup,
    finds the entry whose token sequence is the longest prefix of
    the query. Supports forked conversations: two sessions that
    diverge from a common prefix will both match the parent entry.

    LRU eviction keeps memory bounded to max_entries conversations.
    """

    def __init__(
        self,
        model_name: str = "",
        max_entries: int = 16,
    ):
        self.model_name = model_name
        self.max_entries = max_entries
        self._entries: list[_CacheEntry] = []

    def put(
        self,
        token_ids: list[int],
        prompt_cache: list[Any],
    ) -> None:
        """Store cache state for a token sequence.

        If an existing entry has the same tokens (exact match), it is
        replaced. Otherwise a new entry is added.
        """
        if not token_ids:
            return

        layer_states = [cache_obj.state for cache_obj in prompt_cache]

        # Check for exact match (same conversation, updated)
        for i, entry in enumerate(self._entries):
            if entry.token_ids == token_ids:
                entry.layer_states = layer_states
                entry.last_access = time.time()
                logger.info(
                    "Cache: updated existing entry (%d tokens)", len(token_ids)
                )
                return

        # Check if this extends an existing entry (same conversation, grew)
        # Replace the parent entry since its cache is a subset of the new one
        best_prefix_idx = -1
        best_prefix_len = 0
        for i, entry in enumerate(self._entries):
            n = len(entry.token_ids)
            if n < len(token_ids) and token_ids[:n] == entry.token_ids:
                if n > best_prefix_len:
                    best_prefix_len = n
                    best_prefix_idx = i

        if best_prefix_idx >= 0:
            self._entries[best_prefix_idx] = _CacheEntry(
                token_ids=list(token_ids),
                layer_states=layer_states,
            )
            logger.info(
                "Cache: extended conversation %d -> %d tokens",
                best_prefix_len, len(token_ids),
            )
            return

        # New conversation
        self._entries.append(_CacheEntry(
            token_ids=list(token_ids),
            layer_states=layer_states,
        ))
        logger.info("Cache: new conversation (%d tokens)", len(token_ids))
        self._evict_if_needed()

    def get(
        self,
        token_ids: list[int],
    ) -> Optional[Tuple[list[Any], int]]:
        """Find the longest cached prefix for the given tokens.

        Returns (layer_states, num_matched_tokens) or None if no match.
        """
        if not token_ids:
            return None

        best_entry = None
        best_len = 0

        for entry in self._entries:
            n = len(entry.token_ids)
            # Entry must be a prefix of (or equal to) the query
            if n <= len(token_ids):
                match = token_ids[:n] == entry.token_ids
                if not match and n > 0 and logger.isEnabledFor(logging.DEBUG):
                    for j in range(min(n, len(token_ids))):
                        if entry.token_ids[j] != token_ids[j]:
                            logger.debug(
                                "Cache: prefix mismatch at token %d: "
                                "stored=%d, query=%d (stored_len=%d, query_len=%d)",
                                j, entry.token_ids[j], token_ids[j], n, len(token_ids),
                            )
                            break
                if match and n > best_len:
                    best_len = n
                    best_entry = entry

        if best_entry is None:
            return None

        best_entry.last_access = time.time()
        logger.info(
            "Cache: hit %d/%d tokens cached, %d new",
            best_len, len(token_ids), len(token_ids) - best_len,
        )
        return best_entry.layer_states, best_len

    def reconstruct_cache(
        self,
        layer_states: list,
        trim_to: Optional[int] = None,
        cache_template: Optional[list[Any]] = None,
    ) -> list[Any]:
        """Reconstruct a prompt_cache list from stored layer states.

        Args:
            layer_states: Per-layer state snapshots from get().
            trim_to: If set, trim each layer's KV state to this many
                tokens. Used when the stored cache has more state than
                the prefix match length (e.g. cache includes generated
                tokens beyond the matched prefix).
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

            if trim_to is not None and state is not None:
                state = _trim_state(state, trim_to)

            cache_obj.state = state
            result.append(cache_obj)
        return result

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            # Remove least recently accessed
            oldest_idx = min(
                range(len(self._entries)),
                key=lambda i: self._entries[i].last_access,
            )
            evicted = self._entries.pop(oldest_idx)
            logger.info(
                "Cache: evicted entry (%d tokens, %.1fs old)",
                len(evicted.token_ids), time.time() - evicted.last_access,
            )

    def save_to_disk(self, cache_dir: Path) -> int:
        """Serialize all cached entries to disk.

        Writes one safetensors file per entry plus a JSON index.
        Returns number of entries saved.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "model_name": self.model_name,
            "entries": [],
        }
        saved = 0

        for i, entry in enumerate(self._entries):
            entry_hash = hashlib.sha256(
                str(entry.token_ids[:64]).encode()
            ).hexdigest()[:16]
            entry_file = cache_dir / f"entry_{i}_{entry_hash}.safetensors"

            arrays = {}
            metadata = {}
            for layer_idx, state in enumerate(entry.layer_states):
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
                mx.save_safetensors(str(entry_file), arrays, metadata)

            index["entries"].append({
                "file": entry_file.name,
                "token_ids": entry.token_ids,
                "num_layers": len(entry.layer_states),
            })
            saved += 1

        with open(cache_dir / "cache_index.json", "w") as f:
            json.dump(index, f)

        logger.info("Cache: saved %d entries to %s", saved, cache_dir)
        return saved

    def load_from_disk(self, cache_dir: Path) -> int:
        """Load cached entries from disk. Returns number of entries loaded."""
        index_path = cache_dir / "cache_index.json"
        if not index_path.exists():
            return 0

        with open(index_path) as f:
            index = json.load(f)

        if index.get("model_name") != self.model_name:
            logger.warning(
                "Cache: model mismatch (disk=%s, current=%s), skipping load",
                index.get("model_name"), self.model_name,
            )
            return 0

        loaded = 0
        for entry_info in index["entries"]:
            entry_file = cache_dir / entry_info["file"]
            if not entry_file.exists():
                continue

            arrays, file_metadata = mx.load(
                str(entry_file), return_metadata=True
            )
            num_layers = entry_info["num_layers"]

            layer_states = []
            for layer_idx in range(num_layers):
                if file_metadata.get(f"layer_{layer_idx}_empty") == "1":
                    layer_states.append((None, None))
                    continue
                state = _deserialize_layer_state(
                    arrays, file_metadata, layer_idx
                )
                layer_states.append(state)

            self._entries.append(_CacheEntry(
                token_ids=entry_info["token_ids"],
                layer_states=layer_states,
            ))
            loaded += 1

        logger.info("Cache: loaded %d entries from %s", loaded, cache_dir)
        return loaded

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        return len(self._entries)


def _trim_state(state, length: int):
    """Trim a layer's KV state to the given sequence length.

    Handles plain (keys, values) tuples where keys/values have shape
    (B, H, T, D). Sequence is on axis 2.
    """
    if isinstance(state, (list, tuple)) and len(state) >= 2:
        keys, values = state[0], state[1]
        if isinstance(keys, mx.array) and len(keys.shape) == 4:
            if keys.shape[2] > length:
                keys = keys[:, :, :length, :]
                values = values[:, :, :length, :]
            return (keys, values)
    return state


def _serialize_layer_state(
    arrays: dict, metadata: dict, layer_idx: int, state
) -> None:
    """Serialize a single layer's cache state into arrays dict."""
    if isinstance(state, (list, tuple)) and len(state) >= 2:
        keys, values = state[0], state[1]
        if isinstance(keys, mx.array) and isinstance(values, mx.array):
            metadata[f"layer_{layer_idx}_type"] = "kv"
            arrays[f"layer_{layer_idx}_keys"] = keys
            arrays[f"layer_{layer_idx}_values"] = values
            return
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

    keys = arrays.get(f"layer_{layer_idx}_keys")
    values = arrays.get(f"layer_{layer_idx}_values")
    if keys is not None and values is not None:
        return (keys, values)
    return None


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
