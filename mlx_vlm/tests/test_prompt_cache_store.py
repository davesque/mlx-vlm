import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_vlm.prompt_cache_store import PromptCacheStore, compute_chain_hash


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


class TestChainHash:
    def test_deterministic(self):
        tokens = list(range(50))
        h1 = compute_chain_hash(None, tokens)
        h2 = compute_chain_hash(None, tokens)
        assert h1 == h2
        assert isinstance(h1, bytes)
        assert len(h1) == 32

    def test_different_tokens_different_hash(self):
        h1 = compute_chain_hash(None, list(range(50)))
        h2 = compute_chain_hash(None, list(range(1, 51)))
        assert h1 != h2

    def test_parent_chains(self):
        tokens = list(range(50))
        h_root = compute_chain_hash(None, tokens)
        h_child_a = compute_chain_hash(h_root, tokens)
        h_child_b = compute_chain_hash(None, tokens)
        assert h_child_a != h_child_b

    def test_model_name_isolation(self):
        tokens = list(range(50))
        h1 = compute_chain_hash(None, tokens, model_name="model-a")
        h2 = compute_chain_hash(None, tokens, model_name="model-b")
        assert h1 != h2


class TestPromptCacheStore:
    def _make_ranges_and_tokens(self, *msg_lengths):
        """Helper: create message ranges and token IDs."""
        ranges = []
        pos = 0
        for length in msg_lengths:
            ranges.append((pos, pos + length))
            pos += length
        token_ids = list(range(pos))
        return ranges, token_ids

    def test_store_and_lookup_single_message(self):
        store = PromptCacheStore(model_name="test")
        ranges, tokens = self._make_ranges_and_tokens(50)
        cache = _make_dummy_cache(seq_len=50)

        store.put(ranges, tokens, cache)
        result = store.get(ranges, tokens)
        assert result is not None
        _, num_cached = result
        assert num_cached == 50

    def test_multi_message_lookup(self):
        store = PromptCacheStore(model_name="test")
        # Store 3-message conversation
        ranges, tokens = self._make_ranges_and_tokens(20, 30, 10)
        cache = _make_dummy_cache(seq_len=60)
        store.put(ranges, tokens, cache)

        # Look up same 3 messages: should hit at message 3
        result = store.get(ranges, tokens)
        assert result is not None
        _, num_cached = result
        assert num_cached == 60

    def test_prefix_hit_on_extended_conversation(self):
        store = PromptCacheStore(model_name="test")
        # Store 2-message conversation
        ranges_2, tokens_2 = self._make_ranges_and_tokens(20, 30)
        cache = _make_dummy_cache(seq_len=50)
        store.put(ranges_2, tokens_2, cache)

        # Look up 3-message conversation that starts with same 2 messages
        tokens_3 = list(range(60))  # first 50 same, then 10 new
        ranges_3 = [(0, 20), (20, 50), (50, 60)]
        result = store.get(ranges_3, tokens_3)
        assert result is not None
        _, num_cached = result
        assert num_cached == 50  # cached up to message 2

    def test_shared_system_prompt(self):
        store = PromptCacheStore(model_name="test")
        # Session A: system prompt + user message A
        sys_tokens = list(range(100))
        user_a_tokens = list(range(100, 120))
        ranges_a = [(0, 100), (100, 120)]
        tokens_a = sys_tokens + user_a_tokens
        cache_a = _make_dummy_cache(seq_len=120)
        store.put(ranges_a, tokens_a, cache_a)

        # Session B: same system prompt + different user message
        user_b_tokens = list(range(200, 230))
        ranges_b = [(0, 100), (100, 130)]
        tokens_b = sys_tokens + user_b_tokens
        result = store.get(ranges_b, tokens_b)
        assert result is not None
        _, num_cached = result
        # Should hit at system prompt boundary (message 1), not message 2
        assert num_cached == 100

    def test_miss(self):
        store = PromptCacheStore(model_name="test")
        ranges, tokens = self._make_ranges_and_tokens(50)
        result = store.get(ranges, tokens)
        assert result is None

    def test_lru_eviction(self):
        store = PromptCacheStore(model_name="test", max_entries=2)
        # Fill with 3 entries (triggers eviction of first)
        for i in range(3):
            ranges = [(0, 10)]
            tokens = list(range(i * 100, i * 100 + 10))
            cache = _make_dummy_cache(seq_len=10)
            store.put(ranges, tokens, cache)

        assert store.entry_count == 2

    def test_empty_inputs(self):
        store = PromptCacheStore(model_name="test")
        cache = _make_dummy_cache(seq_len=10)
        assert store.put([], [], cache) == 0
        assert store.get([], []) is None


class TestCacheReconstruction:
    def test_reconstruct_kvcache(self):
        store = PromptCacheStore(model_name="test")
        ranges = [(0, 50)]
        tokens = list(range(50))
        cache = _make_dummy_cache(num_layers=2, seq_len=50)

        for layer_cache in cache:
            k, v = layer_cache.state
            layer_cache.state = (mx.ones_like(k), mx.ones_like(v) * 2)

        store.put(ranges, tokens, cache)
        result = store.get(ranges, tokens)
        assert result is not None
        layer_states, n_tokens = result

        reconstructed = store.reconstruct_cache(layer_states)
        assert len(reconstructed) == 2
        for rc in reconstructed:
            k, v = rc.state
            assert k.shape[2] == 50
            assert mx.allclose(k, mx.ones_like(k))
            assert mx.allclose(v, mx.ones_like(v) * 2)

    def test_trim_to(self):
        store = PromptCacheStore(model_name="test")
        ranges = [(0, 50)]
        tokens = list(range(50))
        cache = _make_dummy_cache(num_layers=2, seq_len=100)  # larger than range

        store.put(ranges, tokens, cache)
        result = store.get(ranges, tokens)
        layer_states, _ = result

        reconstructed = store.reconstruct_cache(layer_states, trim_to=50)
        for rc in reconstructed:
            k, v = rc.state
            assert k.shape[2] == 50


class TestDiskPersistence:
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            store = PromptCacheStore(model_name="test")
            ranges = [(0, 20), (20, 50)]
            tokens = list(range(50))
            cache = _make_dummy_cache(num_layers=2, seq_len=50)
            store.put(ranges, tokens, cache)
            store.save_to_disk(cache_dir)

            store2 = PromptCacheStore(model_name="test")
            store2.load_from_disk(cache_dir)

            result = store2.get(ranges, tokens)
            assert result is not None
            _, n_tokens = result
            assert n_tokens == 50

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
