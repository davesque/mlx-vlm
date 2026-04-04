"""Persistent KV cache store for multi-turn conversation reuse.

Caches KV state at message boundaries using a hash chain (similar to
a blockchain). Each message boundary's hash commits to the entire
conversation history, enabling:

- O(1) prefix lookup via hash dict
- Shared system prompts across sessions (same hash = same state)
- Forked conversations (forks share ancestor entries)
- Per-message granularity (no wasted tokens at block boundaries)

Design:
  Hash(msg_0) = sha256(root + tokens_of_msg_0)
  Hash(msg_0, msg_1) = sha256(Hash(msg_0) + tokens_of_msg_1)
  ...

Each hash maps to the KV cache state at that point in the conversation.
On lookup, we walk the new request's message chain and find the deepest
hash that exists in the store.
"""

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

logger = logging.getLogger(__name__)


def compute_model_fingerprint(model_path: str) -> Optional[str]:
    """Compute a fast fingerprint from model file metadata (no content reads).

    Hashes (filename, size, mtime_ns) for all safetensor files. This
    catches re-downloads, re-quantizations, and model swaps without
    reading any file content.
    """
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return None
    safetensors = sorted(model_dir.glob("*.safetensors"))
    if not safetensors:
        return None
    hasher = hashlib.sha256()
    for f in safetensors:
        st = f.stat()
        hasher.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return hasher.hexdigest()[:16]


def compute_chain_hash(
    parent_hash: Optional[bytes],
    token_ids: List[int],
    model_name: Optional[str] = None,
) -> bytes:
    """Compute a chain hash for a message boundary.

    Each hash commits to the full conversation history via the parent
    chain, just like a block in a blockchain.
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
class _CacheEntry:
    """KV cache state at a message boundary."""

    chain_hash: bytes
    layer_states: list  # per-layer cache.state snapshots (for disk serialization)
    token_count: int  # total tokens up to this boundary
    prompt_cache: Optional[list] = None  # live cache objects (for in-memory reuse)
    last_access: float = field(default_factory=time.time)


class PromptCacheStore:
    """In-memory KV cache store with message-boundary hash chain.

    Stores KV cache state at each message boundary in a conversation.
    Uses a hash chain so that:
    - Shared system prompts hit the same entry across all sessions
    - Forked conversations share all common ancestor entries
    - Lookup is O(num_messages) hash computations + O(1) dict lookups
    - No fixed block size, no wasted tokens at boundaries

    LRU eviction keeps memory bounded.
    """

    def __init__(
        self,
        model_name: str = "",
        max_entries: int = 128,
        kv_bits: Optional[float] = None,
        kv_quant_scheme: Optional[str] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        max_kv_size: Optional[int] = None,
        model_fingerprint: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_entries = max_entries
        # KV cache configuration (used to reject incompatible disk caches)
        self.kv_bits = kv_bits
        self.kv_quant_scheme = kv_quant_scheme
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.max_kv_size = max_kv_size
        self.model_fingerprint = model_fingerprint
        # chain_hash -> _CacheEntry
        self._entries: OrderedDict[bytes, _CacheEntry] = OrderedDict()
        # TurboQuant parameters (auto-detected from first put)
        self._tq_bits: Optional[float] = None
        self._tq_seed: Optional[int] = None

    def put(
        self,
        message_token_ranges: list[tuple[int, int]],
        all_token_ids: list[int],
        prompt_cache: list[Any],
    ) -> int:
        """Store KV cache state at each message boundary.

        Args:
            message_token_ranges: List of (start, end) token index pairs,
                one per message. These define the message boundaries.
            all_token_ids: The full token sequence for the conversation.
            prompt_cache: The live KV cache after generation.

        Returns:
            Number of new entries stored.
        """
        if not message_token_ranges or not all_token_ids:
            return 0

        # Auto-detect TurboQuant parameters from the prompt_cache objects
        if self._tq_bits is None and prompt_cache:
            for cache_obj in prompt_cache:
                if hasattr(cache_obj, "bits") and hasattr(cache_obj, "seed"):
                    self._tq_bits = cache_obj.bits
                    self._tq_seed = cache_obj.seed
                    break

        stored = 0
        parent_hash = None

        for start, end in message_token_ranges:
            msg_tokens = all_token_ids[start:end]
            if not msg_tokens:
                continue

            chain_hash = compute_chain_hash(
                parent_hash, msg_tokens, model_name=self.model_name,
            )

            if chain_hash not in self._entries:
                # Snapshot the full cache state at this boundary
                layer_states = [cache_obj.state for cache_obj in prompt_cache]

                self._entries[chain_hash] = _CacheEntry(
                    chain_hash=chain_hash,
                    layer_states=layer_states,
                    token_count=end,
                    prompt_cache=prompt_cache,
                )
                self._evict_if_needed()
                stored += 1
            else:
                # Touch for LRU
                self._entries.move_to_end(chain_hash)
                self._entries[chain_hash].last_access = time.time()

            parent_hash = chain_hash

        if stored > 0:
            logger.info(
                "Cache: stored %d new message boundaries (%d total entries)",
                stored, len(self._entries),
            )
        return stored

    def get(
        self,
        message_token_ranges: list[tuple[int, int]],
        all_token_ids: list[int],
    ) -> Optional[Tuple[Optional[list[Any]], list[Any], int]]:
        """Find the deepest cached message boundary.

        Walks the hash chain for the given messages and returns the
        cached state at the deepest (most recent) boundary.

        Args:
            message_token_ranges: List of (start, end) token index pairs.
            all_token_ids: The full token sequence.

        Returns:
            (prompt_cache, layer_states, num_cached_tokens) or None.
            prompt_cache is the live cache object list if still in memory,
            or None if loaded from disk (in which case use layer_states
            with reconstruct_cache).
        """
        if not message_token_ranges or not all_token_ids:
            return None

        parent_hash = None
        best_match = None

        for start, end in message_token_ranges:
            msg_tokens = all_token_ids[start:end]
            if not msg_tokens:
                continue

            chain_hash = compute_chain_hash(
                parent_hash, msg_tokens, model_name=self.model_name,
            )

            if chain_hash not in self._entries:
                break

            entry = self._entries[chain_hash]
            self._entries.move_to_end(chain_hash)
            entry.last_access = time.time()
            best_match = entry
            parent_hash = chain_hash

        if best_match is None:
            return None

        logger.info(
            "Cache: hit at %d/%d tokens (%d messages deep, from %s)",
            best_match.token_count,
            max(end for _, end in message_token_ranges),
            sum(1 for _ in self._walk_chain(message_token_ranges, all_token_ids)),
            "memory" if best_match.prompt_cache is not None else "disk",
        )
        return best_match.prompt_cache, best_match.layer_states, best_match.token_count

    def _walk_chain(
        self,
        message_token_ranges: list[tuple[int, int]],
        all_token_ids: list[int],
    ):
        """Yield matching chain hashes (for counting matched depth)."""
        parent_hash = None
        for start, end in message_token_ranges:
            msg_tokens = all_token_ids[start:end]
            if not msg_tokens:
                continue
            chain_hash = compute_chain_hash(
                parent_hash, msg_tokens, model_name=self.model_name,
            )
            if chain_hash not in self._entries:
                break
            yield chain_hash
            parent_hash = chain_hash

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
                the matched boundary (e.g. cache includes generated
                tokens beyond the matched point).
            cache_template: Optional list of empty cache objects to populate.
                If None, creates KVCache objects for each layer.

        Returns:
            List of cache objects with state restored, ready for generate_step.
        """
        from mlx_lm.models.cache import KVCache

        result = []
        for i, state in enumerate(layer_states):
            if _is_turboquant_state(state):
                # TurboQuant states need their own cache type regardless
                # of template (template has plain KVCache for these layers)
                cache_obj = _make_turboquant_cache(
                    state,
                    bits=self._tq_bits,
                    seed=self._tq_seed,
                )
            elif cache_template and i < len(cache_template):
                cache_obj = cache_template[i]
            else:
                cache_obj = KVCache()

            if trim_to is not None and state is not None:
                state = _trim_state(state, trim_to)

            # ArraysCache expects a list, not a tuple
            if isinstance(state, tuple) and hasattr(cache_obj, "cache"):
                state = list(state)

            cache_obj.state = state
            result.append(cache_obj)
        return result

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            evicted_hash, evicted = self._entries.popitem(last=False)
            logger.debug(
                "Cache: evicted entry (%d tokens)", evicted.token_count,
            )

    def save_to_disk(self, cache_dir: Path) -> int:
        """Serialize all cached entries to disk."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "model_name": self.model_name,
            "kv_bits": self.kv_bits,
            "kv_quant_scheme": self.kv_quant_scheme,
            "kv_group_size": self.kv_group_size,
            "quantized_kv_start": self.quantized_kv_start,
            "max_kv_size": self.max_kv_size,
            "model_fingerprint": self.model_fingerprint,
            "entries": [],
        }
        if self._tq_bits is not None:
            index["tq_bits"] = self._tq_bits
        if self._tq_seed is not None:
            index["tq_seed"] = self._tq_seed
        saved = 0

        for chain_hash, entry in self._entries.items():
            hex_hash = chain_hash.hex()
            entry_file = cache_dir / f"{hex_hash[:16]}.safetensors"

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
                "hash": hex_hash,
                "file": entry_file.name,
                "token_count": entry.token_count,
                "num_layers": len(entry.layer_states),
            })
            saved += 1

        with open(cache_dir / "cache_index.json", "w") as f:
            json.dump(index, f)

        logger.info("Cache: saved %d entries to %s", saved, cache_dir)
        return saved

    def load_from_disk(self, cache_dir: Path) -> int:
        """Load cached entries from disk. Returns number loaded."""
        index_path = cache_dir / "cache_index.json"
        if not index_path.exists():
            return 0

        with open(index_path) as f:
            index = json.load(f)

        if index.get("model_name") != self.model_name:
            logger.warning(
                "Cache: model mismatch (disk=%s, current=%s), skipping",
                index.get("model_name"), self.model_name,
            )
            return 0

        # Reject cache if config doesn't match
        config_keys = [
            "kv_bits", "kv_quant_scheme", "kv_group_size",
            "quantized_kv_start", "max_kv_size", "model_fingerprint",
        ]
        mismatches = {}
        for key in config_keys:
            disk_val = index.get(key)
            current_val = getattr(self, key)
            if disk_val != current_val:
                mismatches[key] = (disk_val, current_val)
        if mismatches:
            details = ", ".join(
                f"{k}: disk={d} current={c}" for k, (d, c) in mismatches.items()
            )
            logger.warning("Cache: config mismatch (%s), skipping", details)
            return 0

        # Restore TurboQuant parameters
        if "tq_bits" in index:
            self._tq_bits = index["tq_bits"]
        if "tq_seed" in index:
            self._tq_seed = index["tq_seed"]

        loaded = 0
        for entry_info in index["entries"]:
            hex_hash = entry_info["hash"]
            chain_hash = bytes.fromhex(hex_hash)
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

            self._entries[chain_hash] = _CacheEntry(
                chain_hash=chain_hash,
                layer_states=layer_states,
                token_count=entry_info["token_count"],
            )
            loaded += 1

        logger.info("Cache: loaded %d entries from %s", loaded, cache_dir)
        return loaded

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        return len(self._entries)


def _is_turboquant_state(state) -> bool:
    """Check if a layer state contains TurboQuant NamedTuples."""
    if not isinstance(state, (list, tuple)) or len(state) < 2:
        return False
    keys = state[0]
    return hasattr(keys, '_fields') and 'norms' in getattr(keys, '_fields', ())


def _make_turboquant_cache(state, bits=None, seed=None):
    """Create a TurboQuantKVCache for reconstructing from stored state.

    TurboQuantKVCache.state setter expects (keys_namedtuple, values_namedtuple)
    and handles the offset calculation internally. Codecs are not needed here;
    they rebuild deterministically on the first update_and_fetch call.
    """
    try:
        from mlx_vlm.turboquant import DEFAULT_TURBOQUANT_SEED, TurboQuantKVCache

        if bits is None:
            bits = 3.5
        if seed is None:
            seed = DEFAULT_TURBOQUANT_SEED
        return TurboQuantKVCache(bits=bits, seed=seed)
    except ImportError:
        from mlx_lm.models.cache import KVCache

        return KVCache()


def compute_message_token_ranges(
    processor,
    config,
    messages: list[dict],
    template_kwargs: Optional[dict] = None,
    tools: Optional[list] = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Compute token ranges for each message boundary.

    Tokenizes the conversation at each message prefix to find the
    exact token boundaries. Returns the ranges and full token IDs.

    Args:
        processor: The tokenizer/processor.
        config: Model config (for add_special_tokens logic).
        messages: List of {"role": ..., "content": ...} dicts.
        template_kwargs: Extra kwargs for apply_chat_template.
        tools: Tool definitions to include in the prompt (must match
            what is passed to the formatted_prompt for token counts
            to agree).

    Returns:
        (message_token_ranges, all_token_ids) where ranges is a list
        of (start, end) pairs and all_token_ids is the full sequence.
    """
    from mlx_vlm.server import apply_chat_template

    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )
    add_special = (
        not hasattr(processor, "chat_template")
        if getattr(config, "model_type", "") in ("gemma3", "gemma3n", "gemma4")
        else True
    )

    tkw = template_kwargs or {}
    ranges = []
    prev_end = 0

    for i in range(1, len(messages) + 1):
        prefix_messages = messages[:i]
        try:
            prefix_text = apply_chat_template(
                processor, config, prefix_messages,
                add_generation_prompt=False,
                tools=tools,
                **tkw,
            )
        except Exception:
            # Some templates (e.g. Qwen 3.5) require a user message.
            # Skip this prefix and merge its tokens into the next boundary.
            continue
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=add_special)
        end = len(prefix_tokens)
        ranges.append((prev_end, end))
        prev_end = end

    # Full prompt with generation prompt
    full_text = apply_chat_template(
        processor, config, messages,
        add_generation_prompt=True,
        tools=tools,
        **tkw,
    )
    all_token_ids = tokenizer.encode(full_text, add_special_tokens=add_special)

    return ranges, all_token_ids


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
