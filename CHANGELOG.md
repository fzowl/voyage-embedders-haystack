# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Token-aware batching**: `VoyageDocumentEmbedder` now splits requests by the model's per-request token limit (1M for `voyage-4-lite` and `voyage-3.5-lite`, 320K for `voyage-4` and `voyage-3.5`, 120K for the other models; none is documented for `voyage-code-4`) in addition to `batch_size`. Tokens are counted with the `voyageai` tokenizer, which is downloaded from the Hugging Face Hub on first use. `batch_size` must now be between 1 and 1000, the API limit on inputs per request.
- **Optional auto-chunking** for `VoyageContextualizedDocumentEmbedder` via `enable_auto_chunking=True` (requires `input_type="document"` or `"query"`). Every Document is sent as one whole document with `chunk_size=32000`, so each Document gets one embedding and `source_id` is not used. The default behavior (grouping by `source_id`) is unchanged.
- README: list of supported models and a description of the batching.

### Changed

- The minimum `voyageai` version is now `0.4.0`, which supports auto-chunking.
- Documentation and package description now use the name "Voyage AI by MongoDB". Package, class and module names, import paths and `VOYAGE_API_KEY` are unchanged.
- Docstrings list `voyage-code-4`, `voyage-3.5` and `voyage-3.5-lite` among the models that support `output_dimension` and `output_dtype`, and recommend `rerank-3` and `rerank-3-lite`; the reranker examples use `rerank-3`.

## [3.0.0] - 2026-07-26

### Changed

- **Breaking**: All component `__init__` parameters are now keyword-only. `VoyageTextEmbedder`, `VoyageDocumentEmbedder`, `VoyageContextualizedDocumentEmbedder`, `VoyageMultimodalEmbedder` and `VoyageRanker` no longer accept positional arguments. This keeps the constructors safe to extend and reorder, and matches how components are built from serialized pipelines. The `run()` and `run_async()` signatures are unchanged.

### Migration Notes

- Pass constructor arguments by name:

  ```python
  # Before
  embedder = VoyageTextEmbedder("voyage-4")

  # After
  embedder = VoyageTextEmbedder(model="voyage-4")
  ```

- Pipeline YAML, `from_dict()` and any call that already used keyword arguments are unaffected.

## [2.0.0] - 2026-07-21

### Added

- **Haystack 3.x compatibility**: Components now work with `haystack-ai>=3.0.0`.
- **`warm_up()` method**: All components now support lazy initialization via `warm_up()`. The `client` and `async_client` properties auto-initialize on first access.
- **Async support**: All components now have `async def run_async()` methods for use with Haystack 3.x async pipelines. The synchronous `run()` methods are unchanged.
- **Async client**: Components use `voyageai.AsyncClient` for async operations with `async_embed()`, `async_rerank()`, `async_multimodal_embed()`, and `async_contextualized_embed()`.

### Changed

- **Breaking**: `VoyageDocumentEmbedder.run()` and `VoyageContextualizedDocumentEmbedder.run()` now return new Document instances via `dataclasses.replace()` instead of mutating documents in-place (Haystack 3.x Document immutability).
- **Breaking**: The `client` is no longer created at `__init__`. It is now lazily initialized on first access via `warm_up()`. This means `__init__` no longer fails if the API key is missing — `warm_up()` or the `client`/`async_client` property will raise the error.
- **Breaking**: Minimum Haystack version is now `haystack-ai>=3.0.0`.

### Migration Notes

- Existing synchronous calls to `run()` continue to work unchanged. To run a component asynchronously, call `run_async()` instead:
  ```python
  result = await embedder.run_async(text="Hello")
  ```
- In Haystack pipelines, the `Pipeline.run_async()` method dispatches to each component's `run_async()` automatically.
- If you were checking `embedder.client` at init, note that the client is now lazily created. Call `embedder.warm_up()` if you need it initialized eagerly.

## [1.10.0] - 2026-02-15

### Added

- Support for Voyage 4 model family (`voyage-4`, `voyage-4-large`, `voyage-4-lite`).
- Voyage 4 models support flexible output dimensions (256, 512, 1024, 2048) and multiple output data types (`float`, `int8`, `uint8`, `binary`, `ubinary`).

### Changed

- Updated `VoyageTextEmbedder` and `VoyageDocumentEmbedder` docstrings to document Voyage 4 model support for `output_dimension` and `output_dtype` parameters.
- Updated examples to use `voyage-4` as the default model.

## [1.9.1] - 2026-02-07

### Fixed

- Serialize `chunk_fn` for `VoyageContextualizedDocumentEmbedder` using Haystack's `serialize_callable`/`deserialize_callable`.
- Tighten `chunk_fn` type to `Callable[[str], list[str]]`.
- Add explicit `run()` return types across all components.
- Fix multimodal `_convert_content_item` parameter type.

### Chore

- Add dotenv support to examples and `python-dotenv` to dev dependencies.
- Clean up `pyproject.toml` (remove dead mypy config, redundant ty config).
- Update `CONTRIBUTING.md` with API key setup instructions.

## [1.9.0] - 2026-01-24

### Added

- The new `VoyageMultimodalEmbedder` component supports Voyage's multimodal embedding model (`voyage-multimodal-3.5`).
- Multimodal embeddings can encode text, images, and videos into a shared vector space for cross-modal similarity search.

## [1.8.0] - 2025-11-07

### Added

- The new `VoyageContextualizedDocumentEmbedder` component supports Voyage's contextualized chunk embeddings.
- Contextualized embeddings encode document chunks "in context" with other chunks from the same document, preserving semantic relationships and reducing context loss for improved retrieval accuracy.

## [1.5.0] - 2025-01-22

### Added

- The new `VoyageRanker` component can be used to rerank documents using the `Voyage Reranker` models.
- Matryoshka Embeddings and Quantized Embeddings can now be created using the `output_dimension` and `output_dtype` parameters.

## [1.4.0] - 2024-07-24

### Added

- The maximum timeout and number of retries made by the Client can now be set for the embedders using the `timeout` and `max_retries` parameters.

## [1.3.0] - 2024-03-18

### Changed

- **Breaking Change:** The import path for the embedders has been changed to `haystack_integrations.components.embedders.voyage_embedders`.
- The embedders now use the Haystack `Secret` API for authentication.

## [1.2.0] - 2024-02-02

### Changed

- **Breaking Change:** `VoyageDocumentEmbedder` and `VoyageTextEmbedder` now accept the `model` parameter instead of `model_name`.
- The embedders use the new `voyageai.Client.embed()` method instead of the deprecated `get_embedding` and `get_embeddings` methods.
- Support for the new `truncate` parameter has been added.
- The embedders now return the total number of tokens used as part of the `"total_tokens"` in the metadata.

## [1.1.0] - 2023-12-13

### Added

- Support for `input_type` parameter in `VoyageTextEmbedder` and `VoyageDocumentEmbedder`.

## [1.0.0] - 2023-11-21

### Added

- `VoyageTextEmbedder` and `VoyageDocumentEmbedder` to embed strings and documents.

[unreleased]: https://github.com/awinml/voyage-embedders-haystack/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.10.0...v2.0.0
[1.10.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.9.1...v1.10.0
[1.9.1]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.5.0...v1.8.0
[1.5.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/awinml/voyage-embedders-haystack/releases/tag/v1.0.0
