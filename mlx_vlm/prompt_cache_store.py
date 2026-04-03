"""Persistent KV cache store for multi-turn conversation reuse."""

from __future__ import annotations

import hashlib
from typing import List, Optional

BLOCK_SIZE = 1024


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
