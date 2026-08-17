"""Token-aware batching helpers for the plain-text embedding path.

Voyage's ``embed`` endpoint caps every request by two independent limits: a
maximum number of inputs *and* a maximum number of total tokens per request. The
token cap varies per model, so batching purely by item count can still overflow
the token budget for long inputs. The helpers here split a list of texts into
batches that respect both limits.
"""

# Per-model total-token limits for a single ``embed`` request. Models not listed
# fall back to ``DEFAULT_TOTAL_TOKEN_LIMIT``. Kept in sync with the Voyage AI
# model overview: https://docs.voyageai.com/docs/embeddings
VOYAGE_TOTAL_TOKEN_LIMITS: dict[str, int] = {
    # Contextualized models
    "voyage-context-4": 120_000,
    "voyage-context-3": 120_000,
    # Lite-tier models allow the largest per-request budget
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    # General-purpose flagships
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    # Large / domain / code models
    "voyage-4-large": 120_000,
    "voyage-3-large": 120_000,
    "voyage-code-4": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-2": 120_000,
    "voyage-3-m-exp": 120_000,
}

DEFAULT_TOTAL_TOKEN_LIMIT = 120_000


def resolve_token_limit(model: str) -> int:
    """Return the per-request total-token limit for ``model``.

    Falls back to :data:`DEFAULT_TOTAL_TOKEN_LIMIT` for unknown models so newly
    released models keep working (conservatively) without a code change.
    """
    return VOYAGE_TOTAL_TOKEN_LIMITS.get(model, DEFAULT_TOTAL_TOKEN_LIMIT)


def build_token_aware_batches(
    token_counts: list[int],
    *,
    max_items_per_batch: int,
    max_tokens_per_batch: int,
) -> list[list[int]]:
    """Group input indices into batches bounded by item count and token count.

    :param token_counts:
        Estimated token count for each input, in input order.
    :param max_items_per_batch:
        Maximum number of inputs allowed in a single batch (item-count cap).
    :param max_tokens_per_batch:
        Maximum number of tokens allowed in a single batch.
    :returns:
        A list of batches, each a list of indices into ``token_counts``, in the
        original order. A single input whose token count already exceeds
        ``max_tokens_per_batch`` is placed alone in its own batch and forwarded
        as-is (the API's ``truncation`` flag then governs its handling).
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for index, n_tokens in enumerate(token_counts):
        # If this input would push a non-empty batch over the token budget,
        # close the current batch before starting a new one with this input.
        if current and current_tokens + n_tokens > max_tokens_per_batch:
            batches.append(current)
            current = []
            current_tokens = 0

        current.append(index)
        current_tokens += n_tokens

        # Close the batch when either limit is reached. An oversized single
        # input (n_tokens >= limit) trips the token check and stays alone.
        if len(current) >= max_items_per_batch or current_tokens >= max_tokens_per_batch:
            batches.append(current)
            current = []
            current_tokens = 0

    if current:
        batches.append(current)

    return batches
