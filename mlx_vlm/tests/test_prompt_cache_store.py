import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_vlm.prompt_cache_store import PromptCacheStore


def _make_dummy_cache(num_layers: int = 2, seq_len: int = 100) -> list:
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
        tokens = list(range(50))
        cache = _make_dummy_cache(seq_len=50)

        store.put(tokens, cache)

        result = store.get(tokens)
        assert result is not None
        matched_states, num_matched_tokens = result
        assert num_matched_tokens == 50
        assert len(matched_states) == 2

    def test_lookup_prefix_match(self):
        store = PromptCacheStore(model_name="test")
        tokens_short = list(range(100))
        cache = _make_dummy_cache(seq_len=100)
        store.put(tokens_short, cache)

        # Look up with longer sequence (first 100 should match)
        tokens_long = list(range(200))
        result = store.get(tokens_long)
        assert result is not None
        _, num_matched = result
        assert num_matched == 100

    def test_lookup_miss(self):
        store = PromptCacheStore(model_name="test")
        tokens = list(range(50))
        result = store.get(tokens)
        assert result is None

    def test_forked_conversations(self):
        store = PromptCacheStore(model_name="test")
        # Parent conversation
        parent_tokens = list(range(100))
        cache = _make_dummy_cache(seq_len=100)
        store.put(parent_tokens, cache)

        # Fork A: same prefix, different suffix
        fork_a = list(range(100)) + [1000, 1001, 1002]
        result = store.get(fork_a)
        assert result is not None
        _, num_matched = result
        assert num_matched == 100

        # Fork B: same prefix, different suffix
        fork_b = list(range(100)) + [2000, 2001, 2002]
        result = store.get(fork_b)
        assert result is not None
        _, num_matched = result
        assert num_matched == 100

    def test_extends_existing_conversation(self):
        store = PromptCacheStore(model_name="test")
        tokens_v1 = list(range(50))
        cache_v1 = _make_dummy_cache(seq_len=50)
        store.put(tokens_v1, cache_v1)

        # Same conversation grew
        tokens_v2 = list(range(100))
        cache_v2 = _make_dummy_cache(seq_len=100)
        store.put(tokens_v2, cache_v2)

        # Should have replaced, not duplicated
        assert store.entry_count == 1

        # The longer version should be found
        result = store.get(tokens_v2)
        assert result is not None
        _, num_matched = result
        assert num_matched == 100

    def test_lru_eviction(self):
        store = PromptCacheStore(model_name="test", max_entries=1)
        tokens_a = list(range(50))
        tokens_b = list(range(100, 150))
        cache_a = _make_dummy_cache(seq_len=50)
        cache_b = _make_dummy_cache(seq_len=50)

        store.put(tokens_a, cache_a)
        store.put(tokens_b, cache_b)

        assert store.get(tokens_a) is None
        assert store.get(tokens_b) is not None

    def test_empty_tokens(self):
        store = PromptCacheStore(model_name="test")
        cache = _make_dummy_cache(seq_len=10)
        store.put([], cache)
        assert store.entry_count == 0
        assert store.get([]) is None


class TestCacheReconstruction:
    def test_reconstruct_kvcache(self):
        store = PromptCacheStore(model_name="test")
        tokens = list(range(50))
        cache = _make_dummy_cache(num_layers=2, seq_len=50)

        # Fill with recognizable data
        for layer_cache in cache:
            k, v = layer_cache.state
            layer_cache.state = (mx.ones_like(k), mx.ones_like(v) * 2)

        store.put(tokens, cache)
        result = store.get(tokens)
        assert result is not None
        layer_states, n_tokens = result
        assert n_tokens == 50

        reconstructed = store.reconstruct_cache(layer_states)
        assert len(reconstructed) == 2
        for rc in reconstructed:
            k, v = rc.state
            assert k.shape[2] == 50
            assert mx.allclose(k, mx.ones_like(k))
            assert mx.allclose(v, mx.ones_like(v) * 2)


class TestDiskPersistence:
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            store = PromptCacheStore(model_name="test")
            tokens = list(range(100))
            cache = _make_dummy_cache(num_layers=2, seq_len=100)
            store.put(tokens, cache)
            assert store.entry_count == 1
            store.save_to_disk(cache_dir)

            store2 = PromptCacheStore(model_name="test")
            store2.load_from_disk(cache_dir)
            assert store2.entry_count == 1

            result = store2.get(tokens)
            assert result is not None
            _, n_tokens = result
            assert n_tokens == 100

    def test_save_empty_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store = PromptCacheStore(model_name="test")
            store.save_to_disk(cache_dir)
            assert (cache_dir / "cache_index.json").exists()

    def test_load_nonexistent_dir(self):
        store = PromptCacheStore(model_name="test")
        loaded = store.load_from_disk(Path("/nonexistent/path"))
        assert loaded == 0
