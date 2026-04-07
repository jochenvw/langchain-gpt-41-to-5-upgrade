"""LangChain app with GPT-5 via Azure OpenAI.

Supports four modes:
  --mode chat     Simple chat (no BYOD) — baseline connectivity test
  --mode byod     Chat with Azure AI Search "On Your Data" (BYOD, legacy GPT-4.1 only)
  --mode foundry  GPT-5 RAG via Foundry IQ Agent Service + Azure AI Search
  --mode direct   Direct OpenAI SDK call — bypasses LangChain entirely
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from config import settings


# ---------------------------------------------------------------------------
# AzureChatOpenAI subclass that preserves On Your Data citations
# ---------------------------------------------------------------------------

class AzureChatOpenAIWithContext(AzureChatOpenAI):
    """AzureChatOpenAI wrapper that preserves Azure On Your Data context.

    LangChain's default response parsing drops the ``context`` block that
    Azure OpenAI returns for On Your Data (BYOD) requests.  This subclass
    overrides ``_create_chat_result`` to copy ``context`` (which contains
    citations) into the AIMessage's ``additional_kwargs``, making it
    accessible via ``message.additional_kwargs["context"]``.
    """

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)

        # Extract the Azure OYD context block from the raw response.
        # response.choices[0].message.model_extra["context"] holds citations
        # when data_sources are configured.  model_dump() also preserves it.
        context = None
        if isinstance(response, openai.BaseModel):
            msg = getattr(response.choices[0], "message", None) if getattr(response, "choices", None) else None
            if msg is not None:
                context = (getattr(msg, "model_extra", None) or {}).get("context")
        elif isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                context = choices[0].get("message", {}).get("context")

        if context and result.generations:
            msg = result.generations[0].message
            if isinstance(msg, AIMessage):
                msg.additional_kwargs["context"] = context

        return result


# ---------------------------------------------------------------------------
# LangChain client construction
# ---------------------------------------------------------------------------

def build_llm() -> AzureChatOpenAIWithContext:
    """Build the AzureChatOpenAI instance.

    Returns an ``AzureChatOpenAIWithContext`` which behaves identically to
    the upstream ``AzureChatOpenAI`` but also preserves On Your Data
    citations in ``message.additional_kwargs["context"]``.

    Supports both API-key and Entra ID (Azure AD) authentication based on
    the AZURE_OPENAI_AUTH_TYPE environment variable.
    """
    # GPT-5 only supports the default temperature (1); skip for those models.
    is_gpt5 = settings.deployment.startswith("gpt-5")
    kwargs = dict(
        azure_endpoint=settings.azure_endpoint_base,
        api_version=settings.api_version,
        azure_deployment=settings.deployment,
        **({"temperature": 0.7} if not is_gpt5 else {}),
    )

    if settings.auth_type == "entra":
        token_provider = settings.get_azure_ad_token_provider()
        kwargs["azure_ad_token_provider"] = token_provider
        # AzureChatOpenAI still wants an api_key param; use a placeholder
        kwargs["api_key"] = "entra-id-placeholder"
    else:
        kwargs["api_key"] = settings.api_key
        kwargs["default_headers"] = {"api-key": settings.api_key}

    return AzureChatOpenAIWithContext(**kwargs)


# ---------------------------------------------------------------------------
# BYOD data-source configuration (Azure AI Search)
# ---------------------------------------------------------------------------

def get_byod_extra_body() -> dict:
    """Return the extra_body payload for Azure OpenAI On Your Data.

    Supports three auth modes for Azure AI Search:
      - key:  API key authentication
      - rbac: System-assigned managed identity of the Azure OpenAI resource
      - token: Acquire an access token via DefaultAzureCredential and pass it
    """
    search_auth_type = settings.search_auth_type

    if search_auth_type == "key":
        if not settings.search_api_key or settings.search_api_key.startswith("REPLACE"):
            print("WARNING: AZURE_SEARCH_API_KEY is not configured.")
            print("         BYOD mode will likely fail. Set it in .env first.")
        authentication = {
            "type": "api_key",
            "key": settings.search_api_key,
        }
    elif search_auth_type == "token":
        # Acquire an access token for search using the logged-in user's identity
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
        token = credential.get_token("https://search.azure.com/.default")
        authentication = {
            "type": "access_token",
            "access_token": token.token,
        }
    else:
        # Use system-assigned managed identity of the Azure OpenAI resource
        authentication = {
            "type": "system_assigned_managed_identity",
        }

    data_sources = [
        {
            "type": "azure_search",
            "parameters": {
                "endpoint": settings.search_endpoint,
                "index_name": settings.search_index,
                "authentication": authentication,
            },
        }
    ]
    return {"data_sources": data_sources}


# ---------------------------------------------------------------------------
# Interactive chat loop (LangChain)
# ---------------------------------------------------------------------------

def run_chat(mode: str) -> None:
    """Run an interactive chat loop using LangChain."""
    llm = build_llm()
    byod = mode == "byod"
    extra_body = get_byod_extra_body() if byod else None

    label = "BYOD (On Your Data)" if byod else "simple chat"
    print(f"\n=== LangChain Azure OpenAI — {label} ===")
    print(f"Deployment : {settings.deployment}")
    print(f"Endpoint   : {settings.azure_endpoint_base}")
    if byod:
        print(f"Search     : {settings.search_endpoint} / {settings.search_index}")
    print("Type 'quit' to exit.\n")

    system = SystemMessage(content="You are a helpful assistant.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages = [system, HumanMessage(content=user_input)]

        try:
            if extra_body:
                response = llm.invoke(messages, extra_body=extra_body)
            else:
                response = llm.invoke(messages)

            print(f"\nAssistant: {response.content}\n")

            # In BYOD mode the response may contain citations in the context
            if byod:
                context = (response.additional_kwargs or {}).get("context", {})
                citations = context.get("citations") or context.get("documents")
                if citations:
                    print("--- Citations ---")
                    for i, cite in enumerate(citations, 1):
                        title = cite.get("title", cite.get("filepath", "N/A"))
                        content_preview = cite.get("content", "")[:200]
                        print(f"  [{i}] {title}")
                        if content_preview:
                            print(f"      {content_preview}...")
                    print()

        except Exception as exc:
            _handle_error(exc)


# ---------------------------------------------------------------------------
# Foundry IQ + Agent Service — KB retrieval + GPT-5 via AIProjectClient
# ---------------------------------------------------------------------------


class FoundryAgentSession:
    """Foundry IQ session using KB retrieve API + Agent Service GPT-5 client.

    Uses the Foundry IQ knowledge base ``retrieve`` API for agentic retrieval
    (query planning, parallel search, semantic reranking), then passes the
    retrieved context to GPT-5 via the Agent Service's OpenAI-compatible
    Responses API (obtained from ``AIProjectClient``).

    The MCP-based agent path (MCPTool + agent_reference) requires the project
    managed identity to have Search Service Contributor on the search service.
    This direct approach uses the caller's own credentials for retrieval and
    the Agent Service for generation, avoiding that RBAC requirement.
    """

    def __init__(self, model_override: str | None = None):
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        import requests as _requests

        self._credential = DefaultAzureCredential()
        self._model = model_override or settings.foundry_model_deployment

        # Reuse HTTP session for connection pooling
        self._http = _requests.Session()

        # Pre-cache auth token
        self._token = self._credential.get_token("https://search.azure.com/.default")

        # Foundry IQ knowledge base config
        self._search_endpoint = settings.search_endpoint.rstrip("/")
        self._kb_name = os.environ.get(
            "FOUNDRY_KNOWLEDGE_BASE_NAME", "safety-knowledge-base"
        )

        # Agent Service client for GPT-5 Responses API
        self._project_client = AIProjectClient(
            endpoint=settings.foundry_endpoint,
            credential=self._credential,
        )
        self._openai = self._project_client.get_openai_client()

    def _retrieve(self, user_query: str) -> dict:
        """Call the Foundry IQ knowledge base retrieve API.

        Uses ``low`` reasoning effort for faster retrieval while still
        enabling model-based query planning. Limits to top-5 documents.
        """
        # Refresh token if expired
        import time as _time
        if self._token.expires_on - _time.time() < 120:
            self._token = self._credential.get_token("https://search.azure.com/.default")

        url = (
            f"{self._search_endpoint}/knowledgebases/{self._kb_name}"
            f"/retrieve?api-version=2025-11-01-preview"
        )
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_query}],
                }
            ],
            "retrievalReasoningEffort": {"kind": "low"},
            "includeActivity": False,
            "knowledgeSourceParams": [
                {
                    "knowledgeSourceName": os.environ.get(
                        "FOUNDRY_KNOWLEDGE_SOURCE_NAME",
                        "safety-knowledge-base-source",
                    ),
                    "includeReferences": True,
                    "includeReferenceSourceData": True,
                    "rerankerThreshold": 1.5,
                    "kind": "searchIndex",
                }
            ],
        }
        r = self._http.post(
            url,
            headers={
                "Authorization": f"Bearer {self._token.token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        return r.json()

    def query(self, user_query: str) -> dict:
        """Run a Foundry IQ retrieval → GPT-5 generation pipeline.

        Returns a dict with 'response', 'context', and 'citations'.
        """
        import json as _json

        # Step 1: Foundry IQ agentic retrieval
        kb_result = self._retrieve(user_query)

        # Extract context chunks — keep top 5 for speed
        context_parts = []
        for msg in kb_result.get("response", []):
            for block in msg.get("content", []):
                raw = block.get("text", "")
                try:
                    chunks = _json.loads(raw)
                    if isinstance(chunks, list):
                        for chunk in chunks:
                            ref_id = chunk.get("ref_id", "")
                            text = chunk.get("content", "")
                            if text:
                                context_parts.append(f"[{ref_id}] {text}")
                except (_json.JSONDecodeError, TypeError):
                    if raw:
                        context_parts.append(raw)
        context_parts = context_parts[:5]

        # Extract references for citations
        citations = []
        for ref in kb_result.get("references", []):
            src = ref.get("sourceData") or {}
            citations.append({
                "title": src.get("title", ref.get("docKey", "Source")),
                "ref_id": ref.get("id", ""),
                "doc_key": ref.get("docKey", ""),
                "score": ref.get("rerankerScore"),
            })

        context = "\n\n".join(context_parts)

        if not context:
            return {
                "response": "I don't have enough information to answer that question.",
                "context": "",
                "citations": [],
            }

        # Step 2: GPT-5 Responses API with retrieved context
        system_prompt = (
            "You are a knowledgeable safety compliance assistant. "
            "Answer the user's question based ONLY on the retrieved documents below.\n\n"
            "Guidelines:\n"
            "- Provide thorough, well-structured answers with complete sentences.\n"
            "- Use bullet points or numbered lists when listing multiple items.\n"
            "- Include specific details, numbers, and requirements from the documents.\n"
            "- Cite source reference IDs inline (e.g., [0], [1]) after each claim.\n"
            "- If the documents don't fully answer the question, state what is and isn't covered.\n\n"
            f"--- Retrieved Documents ---\n{context}\n--- End Documents ---"
        )

        try:
            # Retry once on content-filter refusals (transient GPT-5 issue)
            for attempt in range(2):
                response = self._openai.responses.create(
                    model=self._model,
                    instructions=system_prompt,
                    input=user_query,
                )
                text = response.output_text or ""
                if text and "cannot assist" not in text.lower():
                    break

            usage = response.usage
            return {
                "response": text,
                "context": context,
                "citations": citations,
                "prompt_tokens": usage.input_tokens if usage else 0,
                "completion_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            }
        except Exception as exc:
            return {
                "response": "",
                "context": context,
                "citations": citations,
                "error": str(exc),
            }

    def cleanup(self) -> None:
        """No-op — no server-side resources to clean up."""
        pass


def query_foundry_rag(query: str) -> dict:
    """Run a GPT-5 RAG query via Foundry IQ retrieval + Responses API.

    Creates a transient session, runs the query, and returns the result.
    For batch usage (evals), use FoundryAgentSession directly to reuse the client.
    """
    session = FoundryAgentSession()
    try:
        return session.query(query)
    finally:
        session.cleanup()


def run_foundry_chat() -> None:
    """Run an interactive chat loop using Foundry IQ + Agent Service."""
    print(f"\n=== GPT-5 RAG — Foundry IQ KB + Agent Service ===")
    print(f"Project    : {settings.foundry_endpoint}")
    print(f"Model      : {settings.foundry_model_deployment}")
    print(f"KB         : {os.environ.get('FOUNDRY_KNOWLEDGE_BASE_NAME', 'safety-knowledge-base')}")
    print("Type 'quit' to exit.\n")

    session = FoundryAgentSession()
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            try:
                result = session.query(user_input)

                if result.get("error"):
                    print(f"\n[ERROR] {result['error']}\n")
                    continue

                print(f"\nAssistant: {result['response']}\n")

                if result.get("citations"):
                    print("--- Citations ---")
                    for i, cite in enumerate(result["citations"], 1):
                        title = cite.get("title", "N/A")
                        url = cite.get("url", "")
                        print(f"  [{i}] {title}")
                        if url:
                            print(f"      {url}")
                    print()

            except Exception as exc:
                _handle_error(exc)
    finally:
        session.cleanup()


# ---------------------------------------------------------------------------
# Direct OpenAI SDK test (no LangChain)
# ---------------------------------------------------------------------------

def test_direct_openai() -> None:
    """Call the Azure OpenAI endpoint directly via the openai SDK.

    Useful for isolating whether an issue is LangChain-specific or at the
    API / network level.
    """
    print("\n=== Direct OpenAI SDK test ===")
    print(f"Endpoint   : {settings.azure_endpoint_base}")
    print(f"Deployment : {settings.deployment}")
    print(f"API Version: {settings.api_version}")

    client_kwargs = dict(
        azure_endpoint=settings.azure_endpoint_base,
        api_version=settings.api_version,
    )
    if settings.auth_type == "entra":
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        credential = DefaultAzureCredential()
        client_kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
    else:
        client_kwargs["api_key"] = settings.api_key
        client_kwargs["default_headers"] = {"api-key": settings.api_key}

    client = AzureOpenAI(**client_kwargs)

    print("\nSending test message: 'Hello, can you hear me?'")
    try:
        create_kwargs = dict(
            model=settings.deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"},
            ],
        )
        # GPT-5 only accepts the default temperature (1)
        if not settings.deployment.startswith("gpt-5"):
            create_kwargs["temperature"] = 0.7

        completion = client.chat.completions.create(**create_kwargs)

        choice = completion.choices[0]
        print(f"\nResponse: {choice.message.content}")
        print(f"\nUsage: prompt={completion.usage.prompt_tokens}, "
              f"completion={completion.usage.completion_tokens}, "
              f"total={completion.usage.total_tokens}")
        print("\n✓ Direct OpenAI SDK call succeeded.")

    except Exception as exc:
        _handle_error(exc)
        print("\n✗ Direct OpenAI SDK call failed.")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _handle_error(exc: Exception) -> None:
    """Print a user-friendly error message."""
    exc_type = type(exc).__name__
    print(f"\n[ERROR] {exc_type}: {exc}")

    msg = str(exc).lower()
    if "401" in msg or "unauthorized" in msg or "invalid" in msg:
        print("  → Check AZURE_OPENAI_API_KEY in your .env file.")
    elif "404" in msg or "not found" in msg or "resource" in msg:
        print("  → Check AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT.")
        print("    The URL the SDK constructs may not match the APIM route.")
    elif "connection" in msg or "resolve" in msg or "timeout" in msg:
        print("  → Network issue. Check VPN / firewall / endpoint URL.")
    elif "search" in msg or "data_source" in msg:
        print("  → BYOD / data-source error. Check Azure AI Search config.")

    if "--verbose" in sys.argv or "-v" in sys.argv:
        traceback.print_exc()
    else:
        print("  (run with -v for full traceback)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangChain baseline app — Azure OpenAI with optional BYOD or Foundry IQ"
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "byod", "foundry", "direct"],
        default="chat",
        help="chat = simple chat, byod = On Your Data, foundry = Foundry IQ Agent Service, direct = raw OpenAI SDK test",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show full tracebacks on errors",
    )
    args = parser.parse_args()

    if args.mode == "direct":
        test_direct_openai()
    elif args.mode == "foundry":
        run_foundry_chat()
    else:
        run_chat(args.mode)


if __name__ == "__main__":
    main()
