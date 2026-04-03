import pytest
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_vlm.prompt_cache_store import compute_block_hash, PromptCacheStore, BLOCK_SIZE

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


def _make_dummy_cache(num_layers: int = 2, seq_len: int = BLOCK_SIZE) -> list:
    """Create a minimal KVCache list for testing."""
    caches = []
    for _ in range(num_layers):
        c = KVCache()
        keys = mx.zeros((1, 4, seq_len, 64))
        values = mx.zeros((1, 4, seq_len, 64))
        c.update_and_fetch(keys, values)
        caches.append(c)
    return caches


class TestPromptCacheStore:
    def test_store_and_lookup_exact(self):
        store = PromptCacheStore(model_name="test")
        tokens = list(range(BLOCK_SIZE))
        cache = _make_dummy_cache(seq_len=BLOCK_SIZE)

        store.put(tokens, cache)

        result = store.get(tokens)
        assert result is not None
        matched_cache, num_matched_tokens = result
        assert num_matched_tokens == BLOCK_SIZE
        assert len(matched_cache) == 2

    def test_lookup_prefix_match(self):
        store = PromptCacheStore(model_name="test")
        # Store 2 blocks worth of tokens
        tokens_2blocks = list(range(2 * BLOCK_SIZE))
        cache = _make_dummy_cache(seq_len=2 * BLOCK_SIZE)
        store.put(tokens_2blocks, cache)

        # Look up with 3 blocks (only first 2 should match)
        tokens_3blocks = list(range(3 * BLOCK_SIZE))
        result = store.get(tokens_3blocks)
        assert result is not None
        _, num_matched = result
        assert num_matched == 2 * BLOCK_SIZE

    def test_lookup_miss(self):
        store = PromptCacheStore(model_name="test")
        tokens = list(range(BLOCK_SIZE))
        result = store.get(tokens)
        assert result is None

    def test_partial_block_ignored(self):
        store = PromptCacheStore(model_name="test")
        # Fewer than BLOCK_SIZE tokens -> nothing storable
        tokens = list(range(BLOCK_SIZE // 2))
        cache = _make_dummy_cache(seq_len=BLOCK_SIZE // 2)
        store.put(tokens, cache)
        assert store.get(tokens) is None

    def test_lru_eviction(self):
        # max_entries=1 means only 1 block can be stored
        store = PromptCacheStore(model_name="test", max_entries=1)
        tokens_a = list(range(BLOCK_SIZE))
        tokens_b = list(range(BLOCK_SIZE, 2 * BLOCK_SIZE))
        cache_a = _make_dummy_cache(seq_len=BLOCK_SIZE)
        cache_b = _make_dummy_cache(seq_len=BLOCK_SIZE)

        store.put(tokens_a, cache_a)
        store.put(tokens_b, cache_b)

        # First entry evicted
        assert store.get(tokens_a) is None
        assert store.get(tokens_b) is not None


class TestCacheReconstruction:
    def test_reconstruct_kvcache(self):
        """Store a KVCache, retrieve it, use it for continued generation."""
        store = PromptCacheStore(model_name="test")
        tokens = list(range(BLOCK_SIZE))
        cache = _make_dummy_cache(num_layers=2, seq_len=BLOCK_SIZE)

        # Fill cache with recognizable data
        for layer_cache in cache:
            k, v = layer_cache.state
            layer_cache.state = (mx.ones_like(k), mx.ones_like(v) * 2)

        store.put(tokens, cache)
        result = store.get(tokens)
        assert result is not None
        layer_states, n_tokens = result
        assert n_tokens == BLOCK_SIZE

        # Reconstruct cache objects from stored states
        reconstructed = store.reconstruct_cache(layer_states)
        assert len(reconstructed) == 2
        for rc in reconstructed:
            k, v = rc.state
            assert k.shape[2] == BLOCK_SIZE
            # Verify data survived round-trip
            assert mx.allclose(k, mx.ones_like(k))
            assert mx.allclose(v, mx.ones_like(v) * 2)
