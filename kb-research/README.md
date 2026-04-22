# From Azure AI Search Index to Knowledge Base

## Overview

This folder contains scripts and documentation for turning an existing Azure AI Search index into a **knowledge base** for agentic retrieval (Foundry IQ).

**The problem**: Azure's "On Your Data" (BYOD) pipeline has a server-side bug with GPT-5 — it sends `max_tokens` internally, which GPT-5 rejects (see [UPGRADE_NOTES.md](../UPGRADE_NOTES.md)). Knowledge bases offer a **better architecture** that bypasses On Your Data entirely, giving us direct control over retrieval and answer synthesis.

**The goal**: Take an existing Azure AI Search index (no indexer!) and wire it up as a knowledge base that can be queried via the Retrieve API — all with the push model.

### What's in this folder

| File | Purpose |
|------|---------|
| `01_create_index_and_push_data.py` | Creates a search index with semantic configuration and pushes documents |
| `02_create_knowledge_source.py` | Creates a knowledge source that wraps the index |
| `03_create_knowledge_base.py` | Creates a knowledge base and queries it via the Retrieve API |
| `.env.example` | Environment variables template |
| `requirements.txt` | Python dependencies |

---

## What is a Knowledge Base?

A **knowledge base** is a NEW top-level object in Azure AI Search (preview API `2025-11-01-preview`). It is the building block for **Foundry IQ** and agentic RAG scenarios.

What it does:

- **Orchestrates retrieval** from one or more "knowledge sources" (search indexes)
- **Optionally connects to an LLM** (Azure OpenAI) for query planning and answer synthesis
- Provides a single **Retrieve API** endpoint that handles search + reranking + answer generation

> **Note**: This was previously called a "knowledge agent" in earlier preview APIs. It was renamed to "knowledge base" in the `2025-11-01-preview` API version.

Think of it this way:

| Old approach (On Your Data) | New approach (Knowledge Base) |
|----------------------------|-------------------------------|
| Azure manages the full pipeline internally | You control each piece explicitly |
| Server-side bugs (like the GPT-5 `max_tokens` issue) block you | You own the LLM connection — no hidden internal calls |
| Limited configuration | Full control over retrieval instructions, answer formatting, reasoning effort |

---

## Architecture: The Full Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Search Index    │────▶│  Knowledge Source     │────▶│  Knowledge Base  │
│  (with semantic  │     │  (wraps the index,    │     │  (orchestrates   │
│   configuration) │     │   specifies fields    │     │   retrieval +    │
│                  │     │   for citations)      │     │   optional LLM)  │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
                                                      Retrieve API
                                                      (query endpoint)
```

**Data flow:**

1. **You** push documents into a search index (no indexer needed)
2. A **knowledge source** wraps that index — it tells the knowledge base which index to query and which fields to use for citations
3. A **knowledge base** references one or more knowledge sources and connects to an Azure OpenAI model
4. You call the **Retrieve API** on the knowledge base — it searches, reranks, and (optionally) synthesizes an answer using the LLM

---

## Prerequisites

| Requirement | Details |
|------------|---------|
| **Azure AI Search** | Basic tier or higher, with semantic ranker enabled |
| **Azure OpenAI** | A resource with a supported model deployed (gpt-4o, gpt-4.1, gpt-5, etc.) |
| **Python** | 3.12+ |
| **SDK** | `azure-search-documents>=11.6.0b9` (preview) |
| **Auth (RBAC)** | `Search Service Contributor` + `Search Index Data Contributor` on the Search resource, `Cognitive Services User` on the OpenAI resource |
| **Auth (API key)** | Alternatively, API keys for both Search and OpenAI |

Install dependencies:

```bash
cd kb-research
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

---

## Step 1: Create a Search Index with Semantic Configuration

> **Script**: `01_create_index_and_push_data.py`

### What the index needs

Your search index must have:

1. **Searchable string fields** — the text content you want to search over
2. **A semantic configuration** — REQUIRED for knowledge bases (see below)
3. **Citation fields** — title, URL, or other metadata you want returned in results

### What is a semantic configuration?

A **semantic configuration** tells the semantic ranker which fields matter and how to interpret them. It's a section within the index definition — you can add it at any time without rebuilding the index. You can create up to 100 semantic configurations per index.

> **Docs**: [Configure Semantic Ranker](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-configure) · [Semantic Search Overview](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview)

#### Full JSON structure

Here's the complete semantic configuration as it appears in the index definition:

```json
"semantic": {
  "defaultConfiguration": "my-semantic-config",
  "configurations": [
    {
      "name": "my-semantic-config",
      "flightingOptIn": false,
      "rankingOrder": "BoostedRerankerScore",
      "prioritizedFields": {
        "titleField": {
          "fieldName": "title"
        },
        "prioritizedContentFields": [
          {
            "fieldName": "content"
          }
        ],
        "prioritizedKeywordsFields": [
          {
            "fieldName": "category"
          }
        ]
      }
    }
  ]
}
```

#### Top-level settings

| Setting | Type | Purpose |
|---------|------|---------|
| `defaultConfiguration` | string | Name of the semantic configuration to use when none is specified in the query. If your index has only one config, set this so callers don't need to specify it explicitly. Knowledge bases use this to find the right config automatically. |
| `configurations` | array | One or more named semantic configurations (up to 100 per index). |

#### Per-configuration settings

| Setting | Type | Required? | Purpose |
|---------|------|-----------|---------|
| `name` | string | **Yes** | Unique name for this configuration. Referenced by queries and knowledge sources. |
| `prioritizedFields` | object | **Yes** | Defines which fields the semantic ranker uses (see below). |
| `rankingOrder` | string | No | Controls final sort order after semantic reranking. Values: `RerankerScore` (sort by pure semantic score only) or `BoostedRerankerScore` (default — applies scoring profiles on top of semantic reranking). Only matters when you also have a scoring profile with functions. See [Scoring Profiles with Semantic Ranking](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-enable-scoring-profiles). |
| `flightingOptIn` | bool | No | When `true`, opts into prerelease semantic ranking models if one is deployed in your region. Default `false`. Use only in test environments — there's no way to know which model version was used for a given query. |

#### Prioritized fields

The `prioritizedFields` object tells the semantic ranker what each field represents:

| Field | Type | Required? | Purpose |
|-------|------|-----------|---------|
| `titleField` | `{ "fieldName": "..." }` | Recommended | A short string (ideally < 25 words) representing the document title. Used for display and ranking context. Only one title field allowed. |
| `prioritizedContentFields` | `[{ "fieldName": "..." }, ...]` | **REQUIRED** | The main text content — this is what the semantic ranker focuses on for meaning-based reranking. You can list multiple fields **in priority order** — lower-priority fields may be truncated if the combined input exceeds the model's token limit. |
| `prioritizedKeywordsFields` | `[{ "fieldName": "..." }, ...]` | Optional | Tags, categories, or other keyword-like fields. Secondary ranking signal. List in priority order. |

**Field requirements**: All fields assigned to a semantic configuration must be attributed as both `searchable` and `retrievable`. They must be string types (`Edm.String`, `Collection(Edm.String)`, or string subfields of `Edm.ComplexType`).

#### How semantic ranking works (L1 → L2)

1. **L1 — Initial ranking**: Standard BM25 keyword scoring (or RRF for hybrid queries). Produces `@search.score`.
2. **L2 — Semantic reranking**: The semantic model rescores the top ~50 results from L1, using the prioritized fields to understand meaning. Produces `@search.rerankerScore` (range 0–4).
3. **Optional — Scoring profile boost**: If a scoring profile with functions is active and `rankingOrder` is `BoostedRerankerScore`, the profile's boosts are reapplied to the reranked results. Produces `@search.rerankerBoostedScore`.

Knowledge bases use L2 semantic ranking internally — this is why a semantic configuration is **mandatory** for any index used as a knowledge source.

### Why it matters

Without a semantic configuration, you'll get an error when trying to create the knowledge source. The knowledge base relies on semantic reranking to surface the most relevant documents for answer synthesis.

### No indexer needed

We use the **push model** — documents are uploaded directly via `SearchClient.upload_documents()`. No indexer, no data source, no scheduled pulls from blob storage.

### Key code

**Define the index with a semantic configuration:**

```python
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)

fields = [
    SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
    SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
    SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
    SearchField(name="url", type=SearchFieldDataType.String, filterable=True),
    SearchField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
]

semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        content_fields=[SemanticField(field_name="content")],
        keywords_fields=[SemanticField(field_name="category")],
    ),
)

index = SearchIndex(
    name="my-index",
    fields=fields,
    semantic_search=SemanticSearch(
        default_configuration_name="my-semantic-config",  # ← set as default
        configurations=[semantic_config],
    ),
)

index_client.create_or_update_index(index)
```

**Push documents (no indexer):**

```python
from azure.search.documents import SearchClient

search_client = SearchClient(endpoint, index_name, credential)

documents = [
    {"id": "1", "title": "Example Doc", "content": "Full text here...", "url": "https://...", "category": "guide"},
    # ... more documents
]

result = search_client.upload_documents(documents)
print(f"Uploaded {len(result)} documents")
```

---

## Step 2: Create a Knowledge Source

> **Script**: `02_create_knowledge_source.py`

### What is a knowledge source?

A **knowledge source** is a thin wrapper around a search index. It tells the knowledge base:

- **Which index** to query
- **Which fields** to use for citations (`sourceDataFields`)
- Optionally, which semantic configuration to use (overrides the default)

### ⚠️ "Data source" vs "knowledge source" — they are NOT the same

This is a common source of confusion:

| Concept | Used by | Purpose |
|---------|---------|---------|
| **Data source** (`SearchIndexerDataSourceConnection`) | Indexers | Connects to external storage (Blob, SQL, Cosmos DB) so an indexer can pull data |
| **Knowledge source** | Knowledge bases | Wraps a search index so a knowledge base can query it |

They are **completely different objects** in the API. Since we're using the push model (no indexer), we do **NOT** need a data source. We only need a knowledge source.

### Key code

```python
import requests

# The knowledge source API uses the preview REST API
api_version = "2025-11-01-preview"
url = f"{search_endpoint}/knowledgeSources/{ks_name}?api-version={api_version}"

body = {
    "description": "Knowledge source wrapping our search index",
    "type": "azureSearchIndex",
    "azureSearchIndexParameters": {
        "indexName": index_name,
        "semanticConfigurationName": "my-semantic-config",
        "sourceDataFields": {
            "titleFieldName": "title",
            "urlFieldName": "url",
            "contentFieldNames": ["content"],
        },
    },
}

response = requests.put(url, json=body, headers=headers)
response.raise_for_status()
print(f"Knowledge source '{ks_name}' created")
```

> **Note**: The `sourceDataFields` tell the knowledge base which fields to include in citation metadata. This is how the Retrieve API knows to return a title and URL alongside each search result.

---

## Step 3: Create a Knowledge Base (and Query It)

> **Script**: `03_create_knowledge_base.py`

### What the knowledge base does

The knowledge base is the top-level orchestrator. It:

1. **References** one or more knowledge sources
2. **Connects to Azure OpenAI** for query planning and answer synthesis
3. **Exposes the Retrieve API** — a single endpoint for search + reranking + optional LLM answer

### Configuration options

| Setting | Values | Purpose |
|---------|--------|---------|
| `outputMode` | `answerSynthesis` · `extractedData` | `answerSynthesis`: LLM formulates a natural-language answer. `extractedData`: returns raw search results without LLM processing. |
| `retrievalReasoningEffort` | `minimal` · `low` · `medium` | Controls how much LLM processing the knowledge base applies. Higher = better answers but slower and more expensive. |
| `retrievalInstructions` | free-text string | Guides the LLM on which knowledge source to use when, and how to approach multi-source queries. |
| `answerInstructions` | free-text string | Shapes the format and style of the synthesized answer (e.g., "Answer in bullet points", "Cite sources"). |

### Key code

**Create the knowledge base:**

```python
api_version = "2025-11-01-preview"
url = f"{search_endpoint}/knowledgeBases/{kb_name}?api-version={api_version}"

body = {
    "description": "Knowledge base for agentic retrieval",
    "knowledgeSources": [ks_name],
    "models": [
        {
            "kind": "azureOpenAI",
            "azureOpenAIParameters": {
                "resourceUri": openai_endpoint,
                "deploymentId": openai_deployment,
                "modelName": openai_deployment,
                # Auth: either API key or Entra ID (managed identity)
            },
        }
    ],
    "retrievalReasoningEffort": "low",
}

response = requests.put(url, json=body, headers=headers)
response.raise_for_status()
print(f"Knowledge base '{kb_name}' created")
```

**Query via the Retrieve API:**

```python
retrieve_url = f"{search_endpoint}/knowledgeBases/{kb_name}/retrieve?api-version={api_version}"

query_body = {
    "messages": [
        {"role": "user", "content": "What is the refund policy?"}
    ],
    "outputMode": "answerSynthesis",
}

response = requests.post(retrieve_url, json=query_body, headers=headers)
result = response.json()

# The response includes:
# - result["response"]["message"]["content"]  →  the synthesized answer
# - result["response"]["citations"]            →  source documents with title, URL, content
print(result["response"]["message"]["content"])
```

---

## FAQ / Common Questions

### Is a semantic configuration required?

**YES.** Knowledge bases use level 2 (L2) semantic ranking internally. An index without a semantic configuration cannot be used as a knowledge source. You'll get an error at knowledge source creation time. See [Step 1](#step-1-create-a-search-index-with-semantic-configuration) for the full JSON structure and all configuration options.

### Do I need an indexer?

**NO.** You can push documents directly using `SearchClient.upload_documents()`. An indexer is only needed if you want Azure to automatically pull data from external sources (Blob Storage, SQL, Cosmos DB, etc.) on a schedule.

### Do I need a "data source"?

**NO.** A "data source" (`SearchIndexerDataSourceConnection`) is an indexer concept — it defines *where* to pull data from. Since we're using the push model, we skip both the data source and the indexer entirely.

What you **DO** need is a "knowledge source" — a completely different concept that wraps your search index for use by a knowledge base.

### Do I need vector fields?

**Not strictly required**, but recommended for production workloads. Vector fields enable similarity search (finding results by meaning even when keywords don't match). If you add vector fields, you'll also need:

- A vectorizer on the index (to generate embeddings at query time)
- An embedding model deployment (e.g., `text-embedding-3-large`)

For getting started, text fields with semantic ranking are sufficient.

### What API version do I need?

- **Knowledge base / knowledge source APIs**: `2025-11-01-preview`
- **Search index / document upload**: Any stable or preview API version works

### What Python SDK version?

`azure-search-documents>=11.6.0b9` (preview). The knowledge base and knowledge source classes may not be available in all preview SDK versions — the scripts in this folder fall back to the REST API directly when needed.

### How does this compare to On Your Data (BYOD)?

| | On Your Data (BYOD) | Knowledge Base |
|---|---|---|
| **Control** | Azure manages the full pipeline | You control each piece |
| **GPT-5 support** | Broken — sends `max_tokens` internally | Works — you own the LLM connection |
| **Multi-index** | Limited | Supports multiple knowledge sources |
| **Answer shaping** | Limited | `answerInstructions` + `retrievalInstructions` |
| **Agentic use** | Not designed for agents | Built for Foundry IQ and agent frameworks |

---

## Cleanup

Delete resources in this order (knowledge base references knowledge source, which references the index):

```python
# 1. Delete the knowledge base first
requests.delete(
    f"{search_endpoint}/knowledgeBases/{kb_name}?api-version={api_version}",
    headers=headers,
)

# 2. Delete the knowledge source
requests.delete(
    f"{search_endpoint}/knowledgeSources/{ks_name}?api-version={api_version}",
    headers=headers,
)

# 3. Delete the search index
index_client.delete_index(index_name)
```

> **Important**: You must delete in this order. Deleting the knowledge source while a knowledge base still references it will fail.

---

## References

- [Create a Knowledge Base — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-create-knowledge-base)
- [Create a Search Index Knowledge Source](https://learn.microsoft.com/en-us/azure/search/agentic-knowledge-source-how-to-search-index)
- [Index Criteria for Agentic Retrieval](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-create-index)
- [Configure Semantic Ranker](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-configure) — how to add/update semantic configurations
- [Semantic Search Overview](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview) — how L2 semantic ranking works
- [Scoring Profiles with Semantic Ranking](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-enable-scoring-profiles) — `rankingOrder` and `BoostedRerankerScore` explained
- [Data Import — Push Model](https://learn.microsoft.com/en-us/azure/search/search-what-is-data-import) — uploading documents without an indexer
- [Foundry IQ Overview](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/concepts/what-is-foundry-iq)
- [REST API: Indexes — Create or Update](https://learn.microsoft.com/en-us/rest/api/searchservice/indexes/create-or-update) — full index schema reference including `SemanticConfiguration`
- [REST API: Knowledge Sources — Create or Update](https://learn.microsoft.com/en-us/rest/api/searchservice/knowledge-sources/create-or-update?view=rest-searchservice-2025-11-01-preview)
