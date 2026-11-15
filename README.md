<h1 align="center"> <a href="https://github.com/awinml/voyage-embedders-haystack"> Voyage Embedders and Rankers - Haystack </a> </h1>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/voyage-embedders-haystack)](https://pypi.org/project/voyage-embedders-haystack/)
[![Downloads](https://img.shields.io/pypi/dm/voyage-embedders-haystack?color=blue&logo=pypi&logoColor=gold)](https://pypi.org/project/voyage-embedders-haystack/)
[![License](https://img.shields.io/github/license/awinml/voyage-embedders-haystack?color=green)](LICENSE)
[![Tests](https://github.com/awinml/voyage-embedders-haystack/workflows/Test/badge.svg)](https://github.com/awinml/voyage-embedders-haystack/actions)
[![Coverage](https://coveralls.io/repos/github/awinml/voyage-embedders-haystack/badge.svg?branch=main)](https://coveralls.io/github/awinml/voyage-embedders-haystack?branch=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/awinml/voyage-embedders-haystack/main.svg)](https://results.pre-commit.ci/latest/github/awinml/voyage-embedders-haystack/main)
[![Types](https://img.shields.io/badge/types-ty-blue.svg)](https://docs.astral.sh/ty/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

Custom components for [Haystack](https://github.com/deepset-ai/haystack) for creating embeddings and reranking documents using the [Voyage AI by MongoDB](https://voyageai.com/) models.

The embedding models of Voyage AI by MongoDB are state-of-the-art in retrieval accuracy. These models outperform top performing embedding models like `intfloat/e5-mistral-7b-instruct` and `OpenAI/text-embedding-3-large` on the [MTEB Benchmark](https://github.com/embeddings-benchmark/mteb).

#### What's New (v3.0.0)

- **Keyword-only arguments**: All component constructors now take keyword-only arguments — use `VoyageTextEmbedder(model="voyage-4")` rather than `VoyageTextEmbedder("voyage-4")`. This keeps constructors safe to extend without silently shifting the meaning of existing call sites.
- **Agent tool example**: A new [example](https://github.com/awinml/voyage-embedders-haystack/blob/main/examples/agent_tool_example.py) wrapping a Voyage retrieve-and-rerank pipeline as a `ComponentTool` for a Haystack `Agent`.

See the full [Changelog](CHANGELOG.md) for all releases.

## Requirements

- Python 3.10 or higher
- [Voyage AI by MongoDB API Key](https://voyageai.com/)

## Installation

```bash
pip install voyage-embedders-haystack
```

## Usage

You can use the Voyage AI by MongoDB embedding models with multiple components:

- **[VoyageTextEmbedder](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/haystack_integrations/components/embedders/voyage_embedders/voyage_text_embedder.py)**: For generating embeddings for queries.
- **[VoyageDocumentEmbedder](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/haystack_integrations/components/embedders/voyage_embedders/voyage_document_embedder.py)**: For creating semantic embeddings for documents in your indexing pipeline.
- **[VoyageContextualizedDocumentEmbedder](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/haystack_integrations/components/embedders/voyage_embedders/voyage_contextualized_document_embedder.py)**: For creating contextualized embeddings where document chunks are embedded together to preserve context and improve retrieval accuracy.
- **[VoyageMultimodalEmbedder](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/haystack_integrations/components/embedders/voyage_embedders/voyage_multimodal_embedder.py)**: For creating multimodal embeddings that can encode text, images, and videos into a shared vector space.

The reranker models can be used with the [VoyageRanker](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/haystack_integrations/components/rankers/voyage/ranker.py) component.

### Multimodal Embeddings

The `VoyageMultimodalEmbedder` uses the multimodal embedding models of Voyage AI by MongoDB (`voyage-multimodal-3.5`) to encode text, images, and videos into a shared vector space. This enables cross-modal similarity search where you can find images using text queries or find related content across different modalities.

**Key features:**

- Supports text, images (PIL Images, ByteStream), and videos
- Inputs can combine multiple modalities (e.g., text + image)
- Variable output dimensions: 256, 512, 1024 (default), 2048
- Recommended model: `voyage-multimodal-3.5` (`voyage-multimodal-3` supports text and images only, with a fixed dimension of 1024)

**Usage example:**

```python
from haystack.dataclasses import ByteStream
from haystack_integrations.components.embedders.voyage_embedders import VoyageMultimodalEmbedder
from voyageai.video_utils import Video

# Text-only embedding
embedder = VoyageMultimodalEmbedder(model="voyage-multimodal-3.5")
result = embedder.run(inputs=[["What is in this image?"]])

# Mixed text and image embedding
image_bytes = ByteStream.from_file_path("image.jpg")
result = embedder.run(inputs=[["Describe this image:", image_bytes]])

# Video embedding
video = Video.from_path("video.mp4", model="voyage-multimodal-3.5")
result = embedder.run(inputs=[["Describe this video:", video]])
```

### Contextualized Chunk Embeddings

The `VoyageContextualizedDocumentEmbedder` uses the contextualized embedding models of Voyage AI by MongoDB to encode document chunks "in context" with other chunks from the same document. This approach preserves semantic relationships between chunks and reduces context loss, leading to improved retrieval accuracy.

**Key features:**

- Documents are grouped by a metadata field (default: `source_id`)
- Chunks from the same source document are embedded together
- Maintains semantic connections between related chunks
- Recommended model: `voyage-context-4` (the component default)
- Optional auto-chunking: with `enable_auto_chunking=True` (and `input_type="document"`), every Document is sent as one whole document and the API chunks it with the maximum chunk size (32000 tokens), so `source_id` is not needed and each Document gets one embedding

For detailed usage examples, see the [contextualized embedder example](https://github.com/awinml/voyage-embedders-haystack/blob/main/examples/contextualized_embedder_example.py).

Once you've selected the suitable component for your specific use case, initialize the component with the model name and Voyage AI by MongoDB API key. You can also
set the environment variable `VOYAGE_API_KEY` instead of passing the API key as an argument.
To get an API key, please see the [Voyage AI by MongoDB website.](https://www.voyageai.com/)

Information about the supported models can be found in the [Voyage AI by MongoDB documentation.](https://docs.voyageai.com/)

### Supported Models

Any model name accepted by the API can be passed as `model`. The current models are:

| Component | Models |
| --- | --- |
| `VoyageTextEmbedder`, `VoyageDocumentEmbedder` | `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3`, `voyage-finance-2`, `voyage-law-2` |
| `VoyageContextualizedDocumentEmbedder` | `voyage-context-4`, `voyage-context-3` |
| `VoyageMultimodalEmbedder` | `voyage-multimodal-3.5`, `voyage-multimodal-3` |
| `VoyageRanker` | `rerank-3`, `rerank-3-lite`, `rerank-2.5`, `rerank-2.5-lite` |

The text models, except `voyage-finance-2` and `voyage-law-2`, produce 1024-dimensional embeddings by default and support the `output_dimension` values 2048, 1024, 512 and 256. `voyage-finance-2` and `voyage-law-2` always produce 1024 dimensions.

### Batching

`VoyageDocumentEmbedder` splits the Documents into requests of at most `batch_size` Documents (up to the API limit of 1000), and also keeps the total number of tokens of each request within the limit of the model: 1M tokens for `voyage-4-lite` and `voyage-3.5-lite`, 320K for `voyage-4` and `voyage-3.5`, and 120K for most other models (no limit is documented for `voyage-code-4`). The tokens are counted with the tokenizer of the `voyageai` package, which is downloaded from the Hugging Face Hub the first time it is used. The embeddings are returned in the order of the Documents.

## Example

You can find all the examples in the [`examples`](https://github.com/awinml/voyage-embedders-haystack/tree/main/examples) folder.

Below is the example Semantic Search pipeline that uses the [Simple Wikipedia](https://huggingface.co/datasets/pszemraj/simple_wikipedia) Dataset from HuggingFace.

Load the dataset:

```python
# Install HuggingFace Datasets using "pip install datasets"
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Import Voyage Embedders
from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder, VoyageTextEmbedder

# Load first 100 rows of the Simple Wikipedia Dataset from HuggingFace
dataset = load_dataset("pszemraj/simple_wikipedia", split="validation[:100]")

docs = [
    Document(
        content=doc["text"],
        meta={
            "title": doc["title"],
            "url": doc["url"],
        },
    )
    for doc in dataset
]
```

Index the documents to the `InMemoryDocumentStore` using the `VoyageDocumentEmbedder` and `DocumentWriter`:

```python
doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
retriever = InMemoryEmbeddingRetriever(document_store=doc_store)
doc_writer = DocumentWriter(document_store=doc_store)

doc_embedder = VoyageDocumentEmbedder(
    model="voyage-4",
    input_type="document",
)
text_embedder = VoyageTextEmbedder(model="voyage-4", input_type="query")

# Indexing Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=doc_embedder, name="DocEmbedder")
indexing_pipeline.add_component(instance=doc_writer, name="DocWriter")
indexing_pipeline.connect("DocEmbedder", "DocWriter")

indexing_pipeline.run({"DocEmbedder": {"documents": docs}})

print(f"Number of documents in Document Store: {len(doc_store.filter_documents())}")
print(f"First Document: {doc_store.filter_documents()[0]}")
print(f"Embedding of first Document: {doc_store.filter_documents()[0].embedding}")
```

Query the Semantic Search Pipeline using the `InMemoryEmbeddingRetriever` and `VoyageTextEmbedder`:

```python
text_embedder = VoyageTextEmbedder(model="voyage-4", input_type="query")

# Query Pipeline
query_pipeline = Pipeline()
query_pipeline.add_component(instance=text_embedder, name="TextEmbedder")
query_pipeline.add_component(instance=retriever, name="Retriever")
query_pipeline.connect("TextEmbedder.embedding", "Retriever.query_embedding")

# Search
results = query_pipeline.run({"TextEmbedder": {"text": "Which year did the Joker movie release?"}})

# Print text from top result
top_result = results["Retriever"]["documents"][0].content
print("The top search result is:")
print(top_result)
```

### Using Voyage Search as an Agent Tool

You can wrap a Voyage retrieve-and-rerank pipeline as a tool and hand it to a Haystack [`Agent`](https://docs.haystack.deepset.ai/docs/agents) with [`ComponentTool`](https://docs.haystack.deepset.ai/docs/componenttool), so the agent decides when to search your corpus and grounds its answers in the retrieved passages. See the [agent tool example](https://github.com/awinml/voyage-embedders-haystack/blob/main/examples/agent_tool_example.py) for a complete, runnable script.

## Contributing

We welcome contributions from the community! Please take a look at our [contributing guide](CONTRIBUTING.md) for more details on how to get started.

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed changes.

## License

`voyage-embedders-haystack` is distributed under the terms of the [Apache-2.0 license](https://github.com/awinml/voyage-embedders-haystack/blob/main/LICENSE).

Maintained by [Ashwin Mathur](https://github.com/awinml).
