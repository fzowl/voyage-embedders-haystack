import asyncio
from dataclasses import replace as dataclass_replace
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm

from haystack_integrations.components._voyage_client_mixin import VoyageClientMixin
from haystack_integrations.components.embedders.voyage_embedders._batching import (
    DEFAULT_MAX_TOKENS_PER_REQUEST,
    MAX_INPUTS_PER_REQUEST,
    MAX_TOKENS_PER_REQUEST,
    split_into_batches,
)

logger = logging.getLogger(__name__)


@component
class VoyageDocumentEmbedder(VoyageClientMixin):
    """
    A component for computing Document embeddings using Voyage AI by MongoDB embedding models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = VoyageDocumentEmbedder(model="voyage-4")

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        input_type: str | None = None,
        truncate: bool = True,
        prefix: str = "",
        suffix: str = "",
        output_dimension: int | None = None,
        output_dtype: str = "float",
        batch_size: int = 32,
        metadata_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        progress_bar: bool = True,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        """
        Create a VoyageDocumentEmbedder component.

        :param model:
            The name of the model to use.
            For more details on the available models,
            see [Voyage Embeddings documentation](https://docs.voyageai.com/embeddings/).
        :param api_key:
            The Voyage AI by MongoDB API key. It can be explicitly provided or automatically read from the
            environment variable
            VOYAGE_API_KEY (recommended).
        :param input_type:
            Type of the input text. This is used to prepend different prompts to the text. For retrieval/search
            purposes, where a "query" is used to search for relevant information among a collection of data, referred
            to as "documents", it is recommended to specify whether your inputs (texts) are intended as queries or
            documents by setting `input_type` to `"query"` or `"document"` , respectively.
            - Defaults to `None`. This means the embedding model directly converts the inputs (texts) into numerical
                vectors. No prompt is added.
            - Can be set to `"query"`. This will prepend the text with, "Represent the query for retrieving
                supporting documents: ".
            - Can be set to `"document"`. For document, the prompt is "Represent the document for retrieval: ".
        :param truncate:
            Whether to truncate the input texts to fit within the context length. Defaults to `True`.
            - If `True`, over-length input texts will be truncated to fit within the context length, before vectorized
                by the embedding model.
            - If `False`, an error will be raised if any given text exceeds the context length.
        :param output_dimension:
            The dimension of the output embedding. Defaults to `None`.
            - Most models only support a single default dimension, used when `output_dimension` is set to `None` (see
            [model embedding dimensions](https://docs.voyageai.com/docs/embeddings#model-choices) for more details).
            - `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3-large`, `voyage-3.5`,
            `voyage-3.5-lite`, and `voyage-code-3` support the following `output_dimension` values: 2048, 1024
            (default), 512, and 256.
        :param output_dtype: The data type for the embeddings to be returned. Defaults to `"float"`.
            Options: "float", "int8", "uint8", "binary", "ubinary". "float" is supported for all models.
            "int8", "uint8", "binary", and "ubinary" are supported by voyage-4-large, voyage-4, voyage-4-lite,
            voyage-code-4, voyage-3-large, voyage-3.5, voyage-3.5-lite, and voyage-code-3.
            Please see the [FAQ](https://docs.voyageai.com/docs/faq#what-is-quantization-and-output-data-types) for
            more details about output data types.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param batch_size:
            Maximum number of Documents to encode in one request. Must be between 1 and 1000, the API limit on
            inputs per request. Requests are also split so that the total number of tokens stays within the
            per-request limit of the model (for example 120K for `voyage-4-large` and 320K for `voyage-4`). The
            tokens are counted with the tokenizer of the `voyageai` SDK, which is downloaded from the Hugging Face
            Hub on first use. For models that are not listed in the Voyage AI by MongoDB documentation the
            smallest documented limit (120K) is used.
        :param metadata_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param progress_bar:
            Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep the logs
            clean.
        :param timeout:
            Timeout for Voyage AI by MongoDB Client calls, if not set it is inferred from the
            `VOYAGE_TIMEOUT` environment variable
            or set to 30.
        :param max_retries:
            Maximum retries to establish contact with Voyage AI by MongoDB if it returns an internal error,
            if not set it is
            inferred from the `VOYAGE_MAX_RETRIES` environment variable or set to 5.
        """
        self.model = model
        self.input_type = input_type
        self.truncate = truncate
        self.prefix = prefix
        self.suffix = suffix
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        if not 0 < batch_size <= MAX_INPUTS_PER_REQUEST:
            msg = f"batch_size must be between 1 and {MAX_INPUTS_PER_REQUEST}, but got {batch_size}"
            raise ValueError(msg)
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

        self._init_client_lifecycle(api_key, timeout, max_retries)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            input_type=self.input_type,
            truncate=self.truncate,
            prefix=self.prefix,
            suffix=self.suffix,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoyageDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: list[Document]) -> list[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key])
                for key in self.metadata_fields_to_embed
                if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _max_tokens_per_request(self) -> int | None:
        """Total-token cap of one request for the model, or `None` if the model has no documented cap."""
        if self.model in MAX_TOKENS_PER_REQUEST:
            return MAX_TOKENS_PER_REQUEST[self.model]
        logger.warning(
            "No per-request token limit is known for model '{model}', using {limit} tokens.",
            model=self.model,
            limit=DEFAULT_MAX_TOKENS_PER_REQUEST,
        )
        return DEFAULT_MAX_TOKENS_PER_REQUEST

    def _plan_batches(self, texts_to_embed: list[str]) -> list[list[str]]:
        """
        Split the texts into consecutive batches that respect `batch_size` and the token limit of the model.
        """
        max_tokens = self._max_tokens_per_request()
        if max_tokens is None:
            token_counts = [0] * len(texts_to_embed)
        else:
            try:
                token_counts = [len(tokens) for tokens in self.client.tokenize(texts_to_embed, self.model)]
            except Exception as e:
                msg = (
                    f"Could not load the tokenizer of model '{self.model}', which is needed to split the request by "
                    "token count. Please check the model name and that the Hugging Face Hub is reachable."
                )
                raise RuntimeError(msg) from e
        batches = split_into_batches(token_counts, max_inputs=self.batch_size, max_tokens=max_tokens)
        return [[texts_to_embed[i] for i in batch] for batch in batches]

    async def _embed_batch(self, texts_to_embed: list[str]) -> tuple[list[list[float] | list[int]], dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """
        batches = await asyncio.to_thread(self._plan_batches, texts_to_embed)
        all_embeddings: list[list[float] | list[int]] = []
        meta: dict[str, Any] = {}
        meta["total_tokens"] = 0
        for batch in tqdm(batches, disable=not self.progress_bar, desc="Calculating embeddings"):
            response = await self.async_client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type,
                truncation=self.truncate,
                output_dtype=self.output_dtype,
                output_dimension=self.output_dimension,
            )
            all_embeddings.extend(response.embeddings)
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    def _embed_batch_sync(self, texts_to_embed: list[str]) -> tuple[list[list[float] | list[int]], dict[str, Any]]:
        batches = self._plan_batches(texts_to_embed)
        all_embeddings: list[list[float] | list[int]] = []
        meta: dict[str, Any] = {}
        meta["total_tokens"] = 0
        for batch in tqdm(batches, disable=not self.progress_bar, desc="Calculating embeddings"):
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type,
                truncation=self.truncate,
                output_dtype=self.output_dtype,
                output_dimension=self.output_dimension,
            )
            all_embeddings.extend(response.embeddings)
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embed a list of Documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            msg = (
                "VoyageDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, meta = self._embed_batch_sync(texts_to_embed=texts_to_embed)

        new_documents = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(dataclass_replace(doc, embedding=list(emb)))

        return {"documents": new_documents, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embed a list of Documents asynchronously.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            msg = (
                "VoyageDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, meta = await self._embed_batch(texts_to_embed=texts_to_embed)

        new_documents = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(dataclass_replace(doc, embedding=list(emb)))

        return {"documents": new_documents, "meta": meta}
