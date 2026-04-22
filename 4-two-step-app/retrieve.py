"""Retrieve step — Foundry IQ knowledge base retrieve API.

Independent of generation. Returns a context string + citations + latency.
Tune via constructor args (no env-var-only knobs):
    kb_name, knowledge_source_name, reranker_threshold,
    reasoning_effort, max_context_chars.
"""

from __future__ import annotations

import json as _json
import os
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Allow `python 4-two-step-app/retrieve.py` to find shared/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import settings


@dataclass
class RetrieveResult:
    context: str
    citations: list[dict]
    latency_ms: float
    raw: dict = field(default_factory=dict)


class Retriever:
    """Foundry IQ knowledge base retrieve client.

    Acquires an Azure AD token for the search endpoint and reuses an HTTP
    session for connection pooling. Token is refreshed when within 120s
    of expiry.
    """

    def __init__(
        self,
        kb_name: str | None = None,
        knowledge_source_name: str | None = None,
        reranker_threshold: float = 1.5,
        reasoning_effort: str = "minimal",   # minimal | low | medium | high
        max_context_chars: int = 15_000,
        intent_kind: str = "semantic",       # "semantic" or "keyword"
    ):
        from azure.identity import DefaultAzureCredential

        self._credential = DefaultAzureCredential()
        self._token = self._credential.get_token("https://search.azure.com/.default")
        self._http = requests.Session()

        self._search_endpoint = settings.search_endpoint.rstrip("/")
        self._kb_name = kb_name or os.environ.get(
            "FOUNDRY_KNOWLEDGE_BASE_NAME", "safety-knowledge-base"
        )
        self._knowledge_source_name = knowledge_source_name or os.environ.get(
            "FOUNDRY_KNOWLEDGE_SOURCE_NAME", "safety-knowledge-base-source"
        )
        self._reranker_threshold = reranker_threshold
        self._reasoning_effort = reasoning_effort
        self._max_context_chars = max_context_chars
        self._intent_kind = intent_kind

    def _refresh_token_if_needed(self) -> None:
        if self._token.expires_on - _time.time() < 120:
            self._token = self._credential.get_token(
                "https://search.azure.com/.default"
            )

    def retrieve(self, query: str) -> RetrieveResult:
        """Call the KB retrieve endpoint and return parsed context + citations."""
        self._refresh_token_if_needed()

        url = (
            f"{self._search_endpoint}/knowledgebases/{self._kb_name}"
            f"/retrieve?api-version=2025-11-01-preview"
        )
        body = {
            "intents": [{"type": self._intent_kind, "search": query}],
            "retrievalReasoningEffort": {"kind": self._reasoning_effort},
            "includeActivity": False,
            "knowledgeSourceParams": [
                {
                    "knowledgeSourceName": self._knowledge_source_name,
                    "includeReferences": True,
                    "includeReferenceSourceData": True,
                    "rerankerThreshold": self._reranker_threshold,
                    "kind": "searchIndex",
                }
            ],
        }

        start = _time.perf_counter()
        r = self._http.post(
            url,
            headers={
                "Authorization": f"Bearer {self._token.token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        kb = r.json()
        latency_ms = round((_time.perf_counter() - start) * 1000, 1)

        context = self._extract_context(kb)
        citations = self._extract_citations(kb)
        return RetrieveResult(context=context, citations=citations,
                              latency_ms=latency_ms, raw=kb)

    def _extract_context(self, kb_result: dict) -> str:
        """Flatten KB response into a context string, capped at max_context_chars."""
        parts: list[str] = []
        total = 0
        for msg in kb_result.get("response", []):
            for block in msg.get("content", []):
                raw = block.get("text", "")
                try:
                    chunks = _json.loads(raw)
                    if isinstance(chunks, list):
                        for chunk in chunks:
                            ref_id = chunk.get("ref_id", "")
                            text = chunk.get("content", "")
                            if not text:
                                continue
                            part = f"[{ref_id}] {text}"
                            if total + len(part) > self._max_context_chars:
                                return "\n\n".join(parts)
                            parts.append(part)
                            total += len(part)
                except (_json.JSONDecodeError, TypeError):
                    if raw:
                        if total + len(raw) > self._max_context_chars:
                            continue
                        parts.append(raw)
                        total += len(raw)
        return "\n\n".join(parts)

    @staticmethod
    def _extract_citations(kb_result: dict) -> list[dict]:
        out = []
        for ref in kb_result.get("references", []):
            src = ref.get("sourceData") or {}
            out.append({
                "title": src.get("title", ref.get("docKey", "Source")),
                "ref_id": ref.get("id", ""),
                "doc_key": ref.get("docKey", ""),
                "score": ref.get("rerankerScore"),
            })
        return out


# ---------------------------------------------------------------------------
# CLI: quick smoke test
# ---------------------------------------------------------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Foundry IQ KB retrieve smoke test")
    parser.add_argument("query", nargs="?", default="What are the safety requirements?")
    parser.add_argument("--reasoning-effort", default="minimal",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--reranker-threshold", type=float, default=1.5)
    parser.add_argument("--max-context-chars", type=int, default=15_000)
    args = parser.parse_args()

    retriever = Retriever(
        reasoning_effort=args.reasoning_effort,
        reranker_threshold=args.reranker_threshold,
        max_context_chars=args.max_context_chars,
    )
    result = retriever.retrieve(args.query)
    print(f"Latency : {result.latency_ms} ms")
    print(f"Citations: {len(result.citations)}")
    print(f"Context  : {len(result.context)} chars")
    print("--- preview ---")
    print(result.context[:800])


if __name__ == "__main__":
    _main()
