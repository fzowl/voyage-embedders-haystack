from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace as dataclass_replace
from typing import Any, cast

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from tqdm import tqdm

from haystack_integrations.components._voyage_client_mixin import VoyageClientMixin


@component
class VoyageContextualizedDocumentEmbedder(VoyageClientMixin):
    """
    A component for computing contextualized Document embeddings using VoyageAI by MongoDB contextualized models.

    Unlike standard embeddings, contextualized embeddings encode document chunks "in context" with other chunks
    from the same parent document, preserving semantic relationships and reducing context loss.

    There are two ways to run the component:

    - **Auto-chunking (default, `enable_auto_chunking=True`)**: each Document is sent to the API as a single string
      and the `contextualized_embed` endpoint chunks it server-side. The endpoint is called with a `list[str]` input,
      `enable_auto_chunking=True` and `chunk_size=32000` (the per-chunk context limit of `voyage-context-4`). This
      requires `input_type="document"`, which is set automatically when `input_type` is left as `None`.
    - **Pre-chunked (`enable_auto_chunking=False`)**: Documents are grouped by a metadata field (default: 'source_id')
      and each group is sent as one document's chunks (`list[list[str]]`). Documents with the same source_id are
      embedded together as a group. Use this when you have already split documents into chunks yourself.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.voyage_embedders import VoyageContextualizedDocumentEmbedder

    docs = [
        Document(content="Introduction to quantum computing."),
        Document(content="Classical computers use binary bits."),
    ]

    # Auto-chunking (default): each Document is chunked server-side.
    embedder = VoyageContextualizedDocumentEmbedder(model="voyage-context-4")
    result = embedder.run(docs)
    print(result['documents'][0].embedding)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "voyage-context-4",
        input_type: str | None = None,
        prefix: str = "",
        suffix: str = "",
        output_dimension: int | None = None,
        output_dtype: str = "float",
        batch_size: int = 32,
        metadata_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        source_id_field: str = "source_id",
        chunk_fn: Callable[[str], list[str]] | None = None,
        enable_auto_chunking: bool = True,
        chunk_size: int = 32000,
        progress_bar: bool = True,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        """
        Create a VoyageContextualizedDocumentEmbedder component.

        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the environment variable
            VOYAGE_API_KEY (recommended).
        :param model:
            The name of the model to use. Defaults to "voyage-context-4".
            For more details, see [Voyage Contextualized Embeddings](https://docs.voyageai.com/docs/contextualized-chunk-embeddings).
        :param input_type:
            Type of the input text. Can be "query", "document", or None.
            - None: No special prompt is added
            - "query": Prepends "Represent the query for retrieving supporting documents: "
            - "document": Prepends "Represent the document for retrieval: "
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param output_dimension:
            The dimension of the output embedding. Defaults to None (1024 for voyage-context-4).
            voyage-context-4 supports: 2048, 1024 (default), 512, and 256.
        :param output_dtype:
            The data type for the embeddings. Defaults to "float".
            Options: "float", "int8", "uint8", "binary", "ubinary".
        :param batch_size:
            Number of inputs to process per API call. With `enable_auto_chunking=True` this is the number of
            Documents per call; with `enable_auto_chunking=False` it is the number of source groups per call, where
            each group contains multiple chunks from the same source document.
        :param metadata_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param source_id_field:
            The metadata field name used to group documents when `enable_auto_chunking` is `False`. Documents with the
            same value in this field will be embedded together as contextualized chunks. Defaults to "source_id".
        :param chunk_fn:
            Optional custom chunking function to apply. If provided, it will be passed to the API. Only supported when
            `enable_auto_chunking` is `False`; the API rejects combining server-side auto-chunking with a custom
            `chunk_fn`.
        :param enable_auto_chunking:
            Whether to let the `contextualized_embed` endpoint chunk each Document server-side. Defaults to `True`.
            When enabled, each Document is sent as a single `list[str]` input, `input_type` defaults to "document",
            and documents do not need a `source_id` metadata field. When disabled, Documents are grouped by
            `source_id_field` and sent pre-chunked as `list[list[str]]`.
        :param chunk_size:
            Target chunk size in tokens for server-side auto-chunking. Defaults to 32000, the per-chunk context
            limit of `voyage-context-4`. Only used when `enable_auto_chunking` is `True`.
        :param progress_bar:
            Whether to show a progress bar. Can be helpful to disable in production deployments.
        :param timeout:
            Timeout for VoyageAI Client calls, if not set it is inferred from the VOYAGE_TIMEOUT
            environment variable or set to 30.
        :param max_retries:
            Maximum retries to establish contact with VoyageAI if it returns an internal error.
            If not set, inferred from VOYAGE_MAX_RETRIES environment variable or set to 5.
        """
        self.model = model
        self.input_type = input_type
        self.prefix = prefix
        self.suffix = suffix
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.source_id_field = source_id_field
        self.chunk_fn = chunk_fn
        self.enable_auto_chunking = enable_auto_chunking
        self.chunk_size = chunk_size

        if enable_auto_chunking and chunk_fn is not None:
            msg = (
                "`chunk_fn` cannot be combined with `enable_auto_chunking=True`. The contextualized_embed endpoint "
                "does not support server-side auto-chunking together with a custom chunking function. Set "
                "`enable_auto_chunking=False` to use `chunk_fn`."
            )
            raise ValueError(msg)

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
            prefix=self.prefix,
            suffix=self.suffix,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
            source_id_field=self.source_id_field,
            chunk_fn=serialize_callable(self.chunk_fn) if self.chunk_fn else None,
            enable_auto_chunking=self.enable_auto_chunking,
            chunk_size=self.chunk_size,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoyageContextualizedDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["api_key"])
        serialized_chunk_fn = init_params.get("chunk_fn")
        if serialized_chunk_fn:
            init_params["chunk_fn"] = deserialize_callable(serialized_chunk_fn)
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

    def _group_documents_by_source(self, documents: list[Document]) -> tuple[dict[str, list[Document]], list[str]]:
        """
        Group documents by their source_id field.

        :returns:
            A tuple of (grouped_documents, source_order) where:
            - grouped_documents: Dictionary mapping source_id to list of documents
            - source_order: List of source_ids in order they were first encountered
        """
        grouped_docs: dict[str, list[Document]] = defaultdict(list)
        source_order: list[str] = []

        for doc in documents:
            source_id = doc.meta.get(self.source_id_field)
            if source_id is None:
                msg = (
                    f"Document is missing the '{self.source_id_field}' metadata field. "
                    f"All documents must have this field to group contextualized chunks. "
                    f"You can change the field name with the 'source_id_field' parameter."
                )
                raise ValueError(msg)

            source_id_str = str(source_id)
            if source_id_str not in grouped_docs:
                source_order.append(source_id_str)
            grouped_docs[source_id_str].append(doc)

        return dict(grouped_docs), source_order

    def _build_contextualized_api_params(self, batch: list[list[str]]) -> dict[str, Any]:
        """Build shared API parameters for a contextualized embed batch."""
        api_params: dict[str, Any] = {
            "inputs": batch,
            "model": self.model,
        }
        if self.input_type is not None:
            api_params["input_type"] = self.input_type
        if self.output_dtype is not None:
            api_params["output_dtype"] = self.output_dtype
        if self.output_dimension is not None:
            api_params["output_dimension"] = self.output_dimension
        if self.chunk_fn is not None:
            api_params["chunk_fn"] = self.chunk_fn
        return api_params

    def _accumulate_contextualized_embeddings(self, all_embeddings: list[list[int | float]], response: Any) -> None:
        """Extend the embeddings list with results from a contextualized embed response."""
        for result in response.results:
            all_embeddings.extend(cast(list[list[int | float]], result.embeddings))

    def _build_auto_contextualized_api_params(self, inputs: list[str]) -> dict[str, Any]:
        """Build API parameters for a server-side auto-chunked contextualized embed call."""
        api_params: dict[str, Any] = {
            "inputs": inputs,
            "model": self.model,
            "enable_auto_chunking": True,
            "chunk_size": self.chunk_size,
            # Auto-chunking requires input_type="document"; honor an explicit value if provided.
            "input_type": self.input_type or "document",
        }
        if self.output_dtype is not None:
            api_params["output_dtype"] = self.output_dtype
        if self.output_dimension is not None:
            api_params["output_dimension"] = self.output_dimension
        return api_params

    def _accumulate_auto_embeddings(self, all_embeddings: list[list[int | float]], response: Any) -> None:
        """
        Append one embedding per input document from an auto-chunked response.

        With ``enable_auto_chunking=True`` each input string maps to one entry in ``response.results`` whose
        ``embeddings`` list holds the server-generated chunk embeddings. Because ``chunk_size`` defaults to the
        model's per-chunk context limit, a typical document yields a single chunk. To keep the one-embedding-per-
        Document contract, the first chunk embedding is used to represent the document.
        """
        for result in response.results:
            chunk_embeddings = cast(list[list[int | float]], result.embeddings)
            all_embeddings.append(chunk_embeddings[0] if chunk_embeddings else [])

    def _embed_auto_sync(self, texts: list[str], batch_size: int) -> tuple[list[list[int | float]], dict[str, Any]]:
        all_embeddings: list[list[int | float]] = []
        meta: dict[str, Any] = {"total_tokens": 0}

        for i in tqdm(
            range(0, len(texts), batch_size),
            disable=not self.progress_bar,
            desc="Calculating contextualized embeddings",
        ):
            batch = texts[i : i + batch_size]
            api_params = self._build_auto_contextualized_api_params(batch)
            response = self.client.contextualized_embed(**api_params)
            self._accumulate_auto_embeddings(all_embeddings, response)
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    async def _embed_auto(self, texts: list[str], batch_size: int) -> tuple[list[list[int | float]], dict[str, Any]]:
        all_embeddings: list[list[int | float]] = []
        meta: dict[str, Any] = {"total_tokens": 0}

        for i in tqdm(
            range(0, len(texts), batch_size),
            disable=not self.progress_bar,
            desc="Calculating contextualized embeddings",
        ):
            batch = texts[i : i + batch_size]
            api_params = self._build_auto_contextualized_api_params(batch)
            response = await self.async_client.contextualized_embed(**api_params)
            self._accumulate_auto_embeddings(all_embeddings, response)
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    def _embed_batch_sync(
        self, grouped_texts: list[list[str]], batch_size: int
    ) -> tuple[list[list[int | float]], dict[str, Any]]:
        all_embeddings: list[list[int | float]] = []
        meta: dict[str, Any] = {"total_tokens": 0}

        for i in tqdm(
            range(0, len(grouped_texts), batch_size),
            disable=not self.progress_bar,
            desc="Calculating contextualized embeddings",
        ):
            batch = grouped_texts[i : i + batch_size]
            api_params = self._build_contextualized_api_params(batch)
            response = self.client.contextualized_embed(**api_params)
            self._accumulate_contextualized_embeddings(all_embeddings, response)
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    async def _embed_batch(
        self, grouped_texts: list[list[str]], batch_size: int
    ) -> tuple[list[list[int | float]], dict[str, Any]]:
        """
        Embed groups of texts using contextualized embeddings.

        :param grouped_texts:
            List of text groups, where each group is a list of related text chunks.
        :param batch_size:
            Number of groups to process at once.
        :returns:
            Tuple of (all_embeddings, metadata) where all_embeddings is a flat list of embeddings
            corresponding to the flattened input texts.
        """
        all_embeddings: list[list[int | float]] = []
        meta: dict[str, Any] = {"total_tokens": 0}

        for i in tqdm(
            range(0, len(grouped_texts), batch_size),
            disable=not self.progress_bar,
            desc="Calculating contextualized embeddings",
        ):
            batch = grouped_texts[i : i + batch_size]
            api_params = self._build_contextualized_api_params(batch)
            response = await self.async_client.contextualized_embed(**api_params)
            self._accumulate_contextualized_embeddings(all_embeddings, response)
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embed a list of Documents using contextualized embeddings.

        With `enable_auto_chunking=True` (default) each Document is chunked server-side and embedded independently.
        With `enable_auto_chunking=False` Documents are grouped by their source_id metadata field and each group is
        embedded together to preserve context between related chunks.

        :param documents:
            Documents to embed. When `enable_auto_chunking=False`, each Document must have a metadata field
            (default: 'source_id') indicating which chunks belong to the same parent document.
        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model
        """
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            msg = (
                "VoyageContextualizedDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

        if not documents:
            return {"documents": [], "meta": {"total_tokens": 0}}

        if self.enable_auto_chunking:
            texts = self._prepare_texts_to_embed(documents)
            embeddings, meta = self._embed_auto_sync(texts, batch_size=self.batch_size)
            enriched = [dataclass_replace(doc, embedding=emb) for doc, emb in zip(documents, embeddings, strict=True)]
            return {"documents": enriched, "meta": meta}

        grouped_docs, source_order = self._group_documents_by_source(documents)

        grouped_texts = []
        original_indices: list[int] = []
        doc_cursor: dict[int, int] = defaultdict(int)
        doc_positions: dict[int, list[int]] = {}
        for i, d in enumerate(documents):
            doc_positions.setdefault(id(d), []).append(i)
        for source_id in source_order:
            docs = grouped_docs[source_id]
            grouped_texts.append(self._prepare_texts_to_embed(docs))
            for d in docs:
                pos = doc_cursor[id(d)]
                original_indices.append(doc_positions[id(d)][pos])
                doc_cursor[id(d)] = pos + 1

        embeddings, meta = self._embed_batch_sync(grouped_texts, batch_size=self.batch_size)

        # When chunk_fn splits a document into more chunks than the original document,
        # the number of embeddings will exceed the number of original indices.
        # Use loose zip to handle both 1:1 and N:1 mappings gracefully.
        enriched = list(documents)
        for emb, idx in zip(embeddings, original_indices, strict=False):
            enriched[idx] = dataclass_replace(enriched[idx], embedding=emb)

        return {"documents": enriched, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embed a list of Documents using contextualized embeddings asynchronously.

        With `enable_auto_chunking=True` (default) each Document is chunked server-side and embedded independently.
        With `enable_auto_chunking=False` Documents are grouped by their source_id metadata field and each group is
        embedded together to preserve context between related chunks.

        :param documents:
            Documents to embed. When `enable_auto_chunking=False`, each Document must have a metadata field
            (default: 'source_id') indicating which chunks belong to the same parent document.
        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model
        """
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            msg = (
                "VoyageContextualizedDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

        if not documents:
            return {"documents": [], "meta": {"total_tokens": 0}}

        if self.enable_auto_chunking:
            texts = self._prepare_texts_to_embed(documents)
            embeddings, meta = await self._embed_auto(texts, batch_size=self.batch_size)
            enriched = [dataclass_replace(doc, embedding=emb) for doc, emb in zip(documents, embeddings, strict=True)]
            return {"documents": enriched, "meta": meta}

        grouped_docs, source_order = self._group_documents_by_source(documents)

        grouped_texts = []
        original_indices: list[int] = []
        doc_cursor: dict[int, int] = defaultdict(int)
        doc_positions: dict[int, list[int]] = {}
        for i, d in enumerate(documents):
            doc_positions.setdefault(id(d), []).append(i)
        for source_id in source_order:
            docs = grouped_docs[source_id]
            grouped_texts.append(self._prepare_texts_to_embed(docs))
            for d in docs:
                pos = doc_cursor[id(d)]
                original_indices.append(doc_positions[id(d)][pos])
                doc_cursor[id(d)] = pos + 1

        embeddings, meta = await self._embed_batch(grouped_texts, batch_size=self.batch_size)

        # When chunk_fn splits a document into more chunks than the original document,
        # the number of embeddings will exceed the number of original indices.
        # Use loose zip to handle both 1:1 and N:1 mappings gracefully.
        enriched = list(documents)
        for emb, idx in zip(embeddings, original_indices, strict=False):
            enriched[idx] = dataclass_replace(enriched[idx], embedding=emb)

        return {"documents": enriched, "meta": meta}
