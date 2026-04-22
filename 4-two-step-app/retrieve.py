"""Stage 1 of the two-step RAG pipeline: Knowledge Base retrieval.

Calls the Azure AI Search KB retrieve API with semantic search.
Returns normalised document chunks and timing metadata.

The retrieve step is model-agnostic — the same context is reused across
all generation models so comparisons are apples-to-apples.
"""

import time
from dataclasses import dataclass

import requests
from azure.identity import AzureCliCredential

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KB_NAME = "safety-knowledge-base"
KB_API_VERSION = "2025-11-01-preview"
TOP_K = 5
MAX_CHARS_PER_DOC = 3_000
MAX_CONTEXT_CHARS = 12_000


@dataclass
class RetrieveResult:
    """Normalised output of a KB retrieve call."""
    documents: list[dict]       # [{title, content, score}]
    context_text: str           # concatenated context for generation
    latency_ms: float


def retrieve(
    query: str,
    search_endpoint: str,
    credential: AzureCliCredential,
    *,
    top_k: int = TOP_K,
) -> RetrieveResult:
    """Call the Azure AI Search KB retrieve API and return normalised documents.

    Uses ``minimal`` reasoning with ``intents`` format to avoid the
    internal model call that is blocked by Azure Policy (disableLocalAuth).
    """
    url = (
        f"{search_endpoint.rstrip('/')}/knowledgebases('{KB_NAME}')"
        f"/retrieve?api-version={KB_API_VERSION}"
    )

    payload = {
        "retrievalReasoningEffort": {"kind": "minimal"},
        "intents": [{"type": "semantic", "search": query}],
    }

    token = credential.get_token("https://search.azure.com/.default").token

    t0 = time.perf_counter()
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    resp.raise_for_status()
    data = resp.json()

    # The KB returns response[].content[].text — each text is a JSON array of chunks
    raw_response = data.get("response", [])
    documents = []
    total_chars = 0

    import json as _json
    for item in raw_response:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []):
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text_blob = part.get("text", "")
            # text_blob may be a JSON-encoded list of {ref_id, content} objects
            try:
                chunks = _json.loads(text_blob) if text_blob.startswith("[") else [{"content": text_blob}]
            except _json.JSONDecodeError:
                chunks = [{"content": text_blob}]

            if not isinstance(chunks, list):
                chunks = [{"content": str(chunks)}]

            for chunk in chunks[:top_k]:
                content = (chunk.get("content") or "")[:MAX_CHARS_PER_DOC]
                if total_chars + len(content) > MAX_CONTEXT_CHARS:
                    content = content[:MAX_CONTEXT_CHARS - total_chars]
                total_chars += len(content)
                documents.append({
                    "title": chunk.get("title") or f"chunk-{chunk.get('ref_id', len(documents))}",
                    "content": content,
                    "score": chunk.get("score"),
                })
                if total_chars >= MAX_CONTEXT_CHARS:
                    break
        if total_chars >= MAX_CONTEXT_CHARS:
            break

    context_text = "\n\n---\n\n".join(
        f"[{d['title']}]\n{d['content']}" for d in documents
    )

    return RetrieveResult(
        documents=documents,
        context_text=context_text,
        latency_ms=round(latency_ms, 1),
    )
