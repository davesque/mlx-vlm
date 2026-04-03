import pytest
from mlx_vlm.prompt_cache_store import compute_block_hash

BLOCK_SIZE = 1024


class TestBlockHash:
    def test_deterministic(self):
        tokens = list(range(BLOCK_SIZE))
        h1 = compute_block_hash(None, tokens)
        h2 = compute_block_hash(None, tokens)
        assert h1 == h2
        assert isinstance(h1, bytes)
        assert len(h1) == 32  # SHA-256

    def test_different_tokens_different_hash(self):
        h1 = compute_block_hash(None, list(range(BLOCK_SIZE)))
        h2 = compute_block_hash(None, list(range(1, BLOCK_SIZE + 1)))
        assert h1 != h2

    def test_parent_hash_chains(self):
        tokens = list(range(BLOCK_SIZE))
        h_root = compute_block_hash(None, tokens)
        h_child_a = compute_block_hash(h_root, tokens)
        h_child_b = compute_block_hash(None, tokens)
        # Same tokens but different parent -> different hash
        assert h_child_a != h_child_b

    def test_model_name_isolation(self):
        tokens = list(range(BLOCK_SIZE))
        h1 = compute_block_hash(None, tokens, model_name="model-a")
        h2 = compute_block_hash(None, tokens, model_name="model-b")
        assert h1 != h2
