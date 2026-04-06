"""LangChain baseline app with GPT-4.1 via Azure OpenAI APIM Gateway.

Supports four modes:
  --mode chat     Simple chat (no BYOD) — baseline connectivity test
  --mode byod     Chat with Azure AI Search "On Your Data" (BYOD)
  --mode foundry  Chat via Foundry IQ Agent Service with Azure AI Search
  --mode direct   Direct OpenAI SDK call — bypasses LangChain entirely
"""

from __future__ import annotations

import argparse
import json
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
# Foundry IQ Agent Service (replaces BYOD for GPT-5)
# ---------------------------------------------------------------------------

def build_foundry_client():
    """Build an AIProjectClient for the Foundry IQ Agent Service."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    if not settings.foundry_endpoint:
        print("ERROR: FOUNDRY_ENDPOINT is not set. Check your .env file.")
        sys.exit(1)

    return AIProjectClient(
        endpoint=settings.foundry_endpoint,
        credential=DefaultAzureCredential(),
    )


def build_foundry_agent(client):
    """Create a Foundry IQ agent with Azure AI Search grounding.

    Returns the agent object. The caller is responsible for deleting the
    agent when done (or reusing it across queries).
    """
    from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType
    from azure.ai.projects.models import ConnectionType

    # Resolve the Azure AI Search connection
    if settings.foundry_search_connection_id:
        conn_id = settings.foundry_search_connection_id
    else:
        conn_id = client.connections.get_default(ConnectionType.AZURE_AI_SEARCH).id

    ai_search = AzureAISearchTool(
        index_connection_id=conn_id,
        index_name=settings.search_index,
        query_type=AzureAISearchQueryType.SIMPLE,
        top_k=5,
    )

    agent = client.agents.create_agent(
        model=settings.foundry_model_deployment,
        name="safety-assistant",
        instructions="You are a helpful safety compliance assistant. Answer questions using the indexed documents.",
        tools=ai_search.definitions,
        tool_resources=ai_search.resources,
    )
    return agent


def query_foundry_agent(client, agent, query: str) -> dict:
    """Send a query to a Foundry IQ agent and return response + context.

    Returns a dict with 'response', 'context', and 'citations' keys.
    """
    from azure.ai.agents.models import MessageRole, ListSortOrder

    thread = client.agents.threads.create()

    client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=query,
    )

    run = client.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    if run.status == "failed":
        return {"response": "", "context": "", "error": run.last_error}

    messages = client.agents.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING,
    )

    # Extract the last assistant message
    response_text = ""
    citations = []
    for msg in messages.data:
        if msg.role == "assistant":
            for content_item in msg.content:
                if hasattr(content_item, "text"):
                    response_text = content_item.text.value
                    # Extract citation annotations
                    if hasattr(content_item.text, "annotations"):
                        for ann in content_item.text.annotations:
                            if hasattr(ann, "url_citation") or hasattr(ann, "uri_citation"):
                                cite = getattr(ann, "url_citation", None) or getattr(ann, "uri_citation", None)
                                if cite:
                                    citations.append({
                                        "title": getattr(cite, "title", ""),
                                        "url": getattr(cite, "url", "") or getattr(cite, "uri", ""),
                                    })

    # Extract search tool output as context
    context = ""
    try:
        run_steps = client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
        for step in run_steps.data:
            if hasattr(step, "step_details") and hasattr(step.step_details, "tool_calls"):
                for tool_call in step.step_details.tool_calls:
                    if hasattr(tool_call, "azure_ai_search"):
                        search_output = tool_call.azure_ai_search
                        if isinstance(search_output, dict):
                            context += search_output.get("output", "")
                        elif hasattr(search_output, "output"):
                            context += str(search_output.output)
    except Exception:
        pass

    # Clean up thread
    try:
        client.agents.threads.delete(thread.id)
    except Exception:
        pass

    return {
        "response": response_text,
        "context": context,
        "citations": citations,
    }


def run_foundry_chat() -> None:
    """Run an interactive chat loop using the Foundry IQ Agent Service."""
    print(f"\n=== Foundry IQ Agent Service — GPT-5 with Azure AI Search ===")
    print(f"Endpoint   : {settings.foundry_endpoint}")
    print(f"Model      : {settings.foundry_model_deployment}")
    print(f"Search idx : {settings.search_index}")
    print("Type 'quit' to exit.\n")

    client = build_foundry_client()
    agent = build_foundry_agent(client)
    print(f"Agent created: {agent.id}\n")

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

            result = query_foundry_agent(client, agent, user_input)

            if result.get("error"):
                print(f"\n[ERROR] {result['error']}\n")
                continue

            print(f"\nAssistant: {result['response']}\n")

            if result.get("citations"):
                print("--- Citations ---")
                for i, cite in enumerate(result["citations"], 1):
                    print(f"  [{i}] {cite.get('title', 'N/A')}")
                print()
    finally:
        try:
            client.agents.delete_agent(agent.id)
            print("Agent cleaned up.")
        except Exception:
            pass


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
        completion = client.chat.completions.create(
            model=settings.deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"},
            ],
            temperature=0.7,
        )

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
        help="chat = simple chat, byod = On Your Data, foundry = Foundry IQ Agent, direct = raw OpenAI SDK test",
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
