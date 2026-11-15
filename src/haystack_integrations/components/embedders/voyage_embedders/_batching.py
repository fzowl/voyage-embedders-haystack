"""Token-aware batching for the Voyage AI by MongoDB ``embed`` endpoint.

A single ``POST /v1/embeddings`` request is limited by two independent caps: the number of inputs (at most 1000) and
the total number of tokens across all inputs (which depends on the model). Splitting by item count alone can overflow
the token cap, so the helpers here split a list of texts by both.
"""

# Maximum number of inputs in one request, for every model.
MAX_INPUTS_PER_REQUEST = 1000

# Total-token cap of one request, per model:
# https://docs.voyageai.com/docs/embeddings (the `input` parameter).
# `None` means the documentation lists no cap for the model, so only MAX_INPUTS_PER_REQUEST applies.
MAX_TOKENS_PER_REQUEST: dict[str, int | None] = {
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    "voyage-4-large": 120_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-code-4": None,
}

# Contextualized embeddings: with auto-chunking, the chunk size is set to the 32000-token maximum, so that a
# document of up to that size is one chunk, and the total tokens of a request may be up to 120K
# (https://docs.voyageai.com/docs/contextualized-chunk-embeddings).
AUTO_CHUNKING_CHUNK_SIZE = 32_000
AUTO_CHUNKING_MAX_TOKENS_PER_REQUEST = 120_000

# Cap used for models that are not listed above: the smallest documented cap, so a request can never be too large.
DEFAULT_MAX_TOKENS_PER_REQUEST = 120_000


def split_into_batches(token_counts: list[int], *, max_inputs: int, max_tokens: int | None) -> list[list[int]]:
    """
    Split inputs into consecutive batches that respect the input-count and total-token caps.

    :param token_counts:
        Token count of each input, in input order.
    :param max_inputs:
        Maximum number of inputs in one batch.
    :param max_tokens:
        Maximum total tokens in one batch, or `None` for no token cap. An input that alone exceeds the cap gets a
        batch of its own, so the API decides how to handle it (see the `truncate` setting).
    :returns:
        Batches of indices into `token_counts`. They are consecutive slices, so concatenating the batches
        restores the original input order.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index, tokens in enumerate(token_counts):
        over_tokens = max_tokens is not None and current_tokens + tokens > max_tokens
        if current and (len(current) >= max_inputs or over_tokens):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(index)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches
