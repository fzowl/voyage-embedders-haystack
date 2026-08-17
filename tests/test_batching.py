import pytest

from haystack_integrations.components.embedders.voyage_embedders._batching import (
    DEFAULT_TOTAL_TOKEN_LIMIT,
    VOYAGE_TOTAL_TOKEN_LIMITS,
    build_token_aware_batches,
    resolve_token_limit,
)


class TestResolveTokenLimit:
    @pytest.mark.unit
    def test_known_models(self):
        assert resolve_token_limit("voyage-4") == 320_000
        assert resolve_token_limit("voyage-4-lite") == 1_000_000
        assert resolve_token_limit("voyage-4-large") == 120_000
        assert resolve_token_limit("voyage-context-4") == 120_000

    @pytest.mark.unit
    def test_unknown_model_falls_back_to_default(self):
        assert resolve_token_limit("voyage-does-not-exist") == DEFAULT_TOTAL_TOKEN_LIMIT
        # Every configured limit is a positive integer.
        assert all(isinstance(v, int) and v > 0 for v in VOYAGE_TOTAL_TOKEN_LIMITS.values())


class TestBuildTokenAwareBatches:
    @pytest.mark.unit
    def test_splits_at_token_boundary(self):
        # Three inputs of 15 tokens each with a 20-token budget: each input must start a new batch
        # because adding a second would exceed the limit.
        token_counts = [15, 15, 15]
        batches = build_token_aware_batches(token_counts, max_items_per_batch=100, max_tokens_per_batch=20)
        assert batches == [[0], [1], [2]]

    @pytest.mark.unit
    def test_packs_until_token_boundary(self):
        # Inputs fit two-per-batch until the token budget is reached.
        token_counts = [8, 8, 8, 8]
        batches = build_token_aware_batches(token_counts, max_items_per_batch=100, max_tokens_per_batch=20)
        # 8+8=16 <= 20, adding a third (24) would overflow -> new batch.
        assert batches == [[0, 1], [2, 3]]

    @pytest.mark.unit
    def test_single_oversized_text_goes_alone(self):
        # A text larger than the whole budget still gets forwarded on its own.
        token_counts = [10, 500, 10]
        batches = build_token_aware_batches(token_counts, max_items_per_batch=100, max_tokens_per_batch=100)
        assert batches == [[0], [1], [2]]

    @pytest.mark.unit
    def test_first_input_oversized_alone(self):
        token_counts = [999]
        batches = build_token_aware_batches(token_counts, max_items_per_batch=100, max_tokens_per_batch=100)
        assert batches == [[0]]

    @pytest.mark.unit
    def test_item_count_cap_respected(self):
        # Token budget is generous; the item-count cap governs. 5 items, cap 2 -> 3 batches.
        token_counts = [1, 1, 1, 1, 1]
        batches = build_token_aware_batches(token_counts, max_items_per_batch=2, max_tokens_per_batch=1_000_000)
        assert batches == [[0, 1], [2, 3], [4]]

    @pytest.mark.unit
    def test_empty_input(self):
        assert build_token_aware_batches([], max_items_per_batch=32, max_tokens_per_batch=100) == []

    @pytest.mark.unit
    def test_both_limits_interact(self):
        # Item cap of 3, token budget of 20. Tokens force a split before the item cap is hit.
        token_counts = [5, 5, 5, 5, 5]
        batches = build_token_aware_batches(token_counts, max_items_per_batch=3, max_tokens_per_batch=20)
        # First batch fills to 4 items? No: 5+5+5+5=20 hits token limit at 4 items, but item cap is 3.
        # Item cap closes the batch at 3 items (15 tokens); next batch takes the rest.
        assert batches == [[0, 1, 2], [3, 4]]
