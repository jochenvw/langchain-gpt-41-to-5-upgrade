"""LangChain baseline app with GPT-4.1 via Azure OpenAI APIM Gateway.

Supports three modes:
  --mode chat    Simple chat (no BYOD) — baseline connectivity test
  --mode byod    Chat with Azure AI Search "On Your Data" (BYOD)
  --mode direct  Direct OpenAI SDK call — bypasses LangChain entirely
"""

import argparse
import json
import sys
import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from config import settings


# ---------------------------------------------------------------------------
# LangChain client construction
# ---------------------------------------------------------------------------

def build_llm() -> AzureChatOpenAI:
    """Build the AzureChatOpenAI instance.

    Supports both API-key and Entra ID (Azure AD) authentication based on
    the AZURE_OPENAI_AUTH_TYPE environment variable.
    """
    kwargs = dict(
        azure_endpoint=settings.azure_endpoint_base,
        api_version=settings.api_version,
        azure_deployment=settings.deployment,
        temperature=0.7,
    )

    if settings.auth_type == "entra":
        token_provider = settings.get_azure_ad_token_provider()
        kwargs["azure_ad_token_provider"] = token_provider
        # AzureChatOpenAI still wants an api_key param; use a placeholder
        kwargs["api_key"] = "entra-id-placeholder"
    else:
        kwargs["api_key"] = settings.api_key
        kwargs["default_headers"] = {"api-key": settings.api_key}

    return AzureChatOpenAI(**kwargs)


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
            if byod and hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                # Azure OYD returns citations under various keys depending
                # on SDK version; try the common ones.
                context = meta.get("context", {})
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
        description="LangChain baseline app — Azure OpenAI GPT-4.1 with optional BYOD"
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "byod", "direct"],
        default="chat",
        help="chat = simple chat, byod = On Your Data, direct = raw OpenAI SDK test",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show full tracebacks on errors",
    )
    args = parser.parse_args()

    if args.mode == "direct":
        test_direct_openai()
    else:
        run_chat(args.mode)


if __name__ == "__main__":
    main()
