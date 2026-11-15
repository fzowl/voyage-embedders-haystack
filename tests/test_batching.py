import pytest

from haystack_integrations.components.embedders.voyage_embedders._batching import split_into_batches


class TestSplitIntoBatches:
    @pytest.mark.unit
    def test_splits_at_token_cap(self):
        assert split_into_batches([40, 40, 40, 40, 40], max_inputs=1000, max_tokens=100) == [[0, 1], [2, 3], [4]]

    @pytest.mark.unit
    def test_batch_may_fill_the_cap_exactly(self):
        assert split_into_batches([50, 50, 50], max_inputs=1000, max_tokens=100) == [[0, 1], [2]]

    @pytest.mark.unit
    def test_splits_at_input_cap(self):
        assert split_into_batches([1] * 5, max_inputs=2, max_tokens=1_000_000) == [[0, 1], [2, 3], [4]]

    @pytest.mark.unit
    def test_oversized_input_gets_its_own_batch(self):
        assert split_into_batches([10, 500, 10], max_inputs=1000, max_tokens=100) == [[0], [1], [2]]

    @pytest.mark.unit
    def test_no_token_cap(self):
        assert split_into_batches([10**9, 10**9, 10**9], max_inputs=2, max_tokens=None) == [[0, 1], [2]]

    @pytest.mark.unit
    def test_order_is_preserved(self):
        counts = [30, 80, 5, 90, 90, 10, 60]
        batches = split_into_batches(counts, max_inputs=3, max_tokens=100)
        assert [i for batch in batches for i in batch] == list(range(len(counts)))
        assert all(sum(counts[i] for i in batch) <= 100 for batch in batches)

    @pytest.mark.unit
    def test_empty(self):
        assert split_into_batches([], max_inputs=10, max_tokens=100) == []
