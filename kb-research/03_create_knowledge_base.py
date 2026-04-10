"""Create and query an Azure AI Search knowledge base (Foundry IQ).

A knowledge base is the top-level agentic retrieval object — it orchestrates
retrieval from one or more knowledge sources and optionally uses an LLM for
query planning and answer synthesis.

Uses the REST API directly (the Python SDK classes for knowledge bases are
not yet generally available).

Prerequisites:
  - A knowledge source already exists (see 02_create_knowledge_source.py)
  - Azure OpenAI deployment accessible to the search service
  - RBAC: caller needs Search Service Contributor to create the KB,
    and the search service's managed identity needs Cognitive Services User
    on the Azure OpenAI resource for query-time LLM calls

Usage:
    # Create KB and run a test query
    uv run python kb-research/03_create_knowledge_base.py

    # Query only (KB already exists)
    uv run python kb-research/03_create_knowledge_base.py --skip-create

    # Create only, no query
    uv run python kb-research/03_create_knowledge_base.py --skip-query

    # Custom names and query
    uv run python kb-research/03_create_knowledge_base.py \\
        --name my-kb --ks-name my-ks \\
        --query "How should chemical spills be handled?"
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root, then from kb-research/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(KB_DIR / ".env", override=True)

API_VERSION = "2025-11-01-preview"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def get_search_headers() -> dict:
    """Build HTTP headers for Azure AI Search REST calls.

    Uses DefaultAzureCredential (bearer token) by default.  Falls back to
    API key if AZURE_SEARCH_AUTH_TYPE=key and AZURE_SEARCH_API_KEY is set.
    """
    auth_type = os.environ.get("AZURE_SEARCH_AUTH_TYPE", "token").lower()
    api_key = os.environ.get("AZURE_SEARCH_API_KEY", "").strip()

    headers = {"Content-Type": "application/json"}

    if auth_type == "key" and api_key:
        headers["api-key"] = api_key
    else:
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
        token = credential.get_token("https://search.azure.com/.default")
        headers["Authorization"] = f"Bearer {token.token}"

    return headers


def get_search_endpoint() -> str:
    """Return the search endpoint, stripping any trailing slash."""
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        print("ERROR: AZURE_SEARCH_ENDPOINT must be set in .env or environment.")
        sys.exit(1)
    return endpoint


# ---------------------------------------------------------------------------
# Build KB payload
# ---------------------------------------------------------------------------

def build_kb_body(name: str, ks_name: str) -> dict:
    """Build the JSON body for creating/updating a knowledge base."""
    openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    openai_auth_type = os.environ.get("AZURE_OPENAI_AUTH_TYPE", "entra").lower()

    if not openai_endpoint or not openai_deployment:
        print("ERROR: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set.")
        sys.exit(1)

    # Strip /openai suffix if present — the API expects the base resource URI
    openai_resource_uri = openai_endpoint.replace("/openai", "")

    model_config: dict = {
        "kind": "azureOpenAI",
        "azureOpenAIParameters": {
            "resourceUri": openai_resource_uri,
            "deploymentId": openai_deployment,
            "modelName": openai_deployment,
        }
    }

    # Only include apiKey when using key-based auth; omit for managed identity
    if openai_auth_type == "key" and openai_api_key:
        model_config["azureOpenAIParameters"]["apiKey"] = openai_api_key

    body = {
        "name": name,
        "description": "Demo knowledge base for the Contoso Energy Safety Manual.",
        "knowledgeSources": [
            {"name": ks_name}
        ],
        "models": [model_config],
        "retrievalReasoningEffort": {"kind": "low"},
        "outputMode": "answerSynthesis",
    }
    return body


# ---------------------------------------------------------------------------
# REST operations
# ---------------------------------------------------------------------------

def list_knowledge_bases() -> list:
    """List all knowledge bases on the search service."""
    import requests

    endpoint = get_search_endpoint()
    headers = get_search_headers()
    url = f"{endpoint}/knowledgebases?api-version={API_VERSION}"

    print("Listing existing knowledge bases...")
    resp = requests.get(url, headers=headers, timeout=30)

    if resp.status_code == 200:
        data = resp.json()
        kbs = data.get("value", [])
        if kbs:
            for kb in kbs:
                print(f"  • {kb.get('name', '?')} — {kb.get('description', '(no description)')}")
        else:
            print("  (none found)")
        return kbs
    else:
        print(f"  Warning: Could not list KBs (HTTP {resp.status_code}): {resp.text[:300]}")
        return []


def create_or_update_kb(name: str, ks_name: str) -> dict | None:
    """Create or update a knowledge base via PUT."""
    import requests

    endpoint = get_search_endpoint()
    headers = get_search_headers()
    headers["Prefer"] = "return=representation"

    url = f"{endpoint}/knowledgebases('{name}')?api-version={API_VERSION}"
    body = build_kb_body(name, ks_name)

    print(f"\nCreating/updating knowledge base '{name}'...")
    print(f"  Knowledge source: {ks_name}")
    print(f"  OpenAI endpoint:  {body['models'][0]['azureOpenAIParameters']['resourceUri']}")
    print(f"  OpenAI model:     {body['models'][0]['azureOpenAIParameters']['deploymentId']}")
    auth_mode = "API key" if "apiKey" in body["models"][0]["azureOpenAIParameters"] else "managed identity"
    print(f"  OpenAI auth:      {auth_mode}")

    resp = requests.put(url, headers=headers, json=body, timeout=60)

    if resp.status_code in (200, 201):
        result = resp.json()
        verb = "Updated" if resp.status_code == 200 else "Created"
        print(f"  ✓ {verb} knowledge base '{result.get('name', name)}'")
        return result
    else:
        print(f"  ✗ Failed (HTTP {resp.status_code})")
        print(f"    {resp.text[:500]}")
        return None


def query_knowledge_base(name: str, query_text: str) -> dict | None:
    """Query a knowledge base using the retrieve endpoint."""
    import requests

    endpoint = get_search_endpoint()
    headers = get_search_headers()

    url = f"{endpoint}/knowledgebases('{name}')/retrieve?api-version={API_VERSION}"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": query_text}],
            }
        ],
        "retrievalReasoningEffort": {"kind": "low"},
    }

    print(f"\nQuerying knowledge base '{name}'...")
    print(f"  Query: {query_text}")

    resp = requests.post(url, headers=headers, json=body, timeout=120)

    if resp.status_code == 200:
        result = resp.json()
        print("  ✓ Query succeeded\n")
        print_query_response(result)
        return result
    else:
        print(f"  ✗ Query failed (HTTP {resp.status_code})")
        print(f"    {resp.text[:500]}")
        return None


# ---------------------------------------------------------------------------
# Pretty-print query response
# ---------------------------------------------------------------------------

def print_query_response(result: dict) -> None:
    """Pretty-print a knowledge base query response."""
    # Dump the full raw response first for debugging
    raw = json.dumps(result, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)

    # Extract answer from the response — structure varies by outputMode
    # answerSynthesis: result.response[0].content[0].text
    # extractedData:   result.response[0].content[0].text (may be raw chunks)
    response_items = result.get("response", [])
    if isinstance(response_items, list):
        for item in response_items:
            content_parts = item.get("content", [])
            if isinstance(content_parts, list):
                for part in content_parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        print(part.get("text", ""))
            elif isinstance(content_parts, str):
                print(content_parts)

    # Fallback: try older/alternative response shapes
    if not response_items:
        answer = result.get("answer", "")
        if answer:
            print(answer)
        else:
            messages = result.get("messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                p.get("text", "") for p in content if isinstance(p, dict)
                            )
                        if content:
                            print(content)

    # Print citations / references if available
    citations = result.get("citations", result.get("references", []))
    if not citations:
        # Try from activity
        activity = result.get("activity", [])
        if isinstance(activity, list):
            for act in activity:
                if isinstance(act, dict):
                    citations = act.get("citations", act.get("references", []))
                    if citations:
                        break
        elif isinstance(activity, dict):
            citations = activity.get("citations", activity.get("references", []))

    if citations and isinstance(citations, list):
        print(f"\n{'─' * 60}")
        print(f"CITATIONS ({len(citations)})")
        print("─" * 60)
        for i, cite in enumerate(citations, 1):
            if not isinstance(cite, dict):
                continue
            title = cite.get("title", cite.get("filepath", cite.get("url", f"Source {i}")))
            chunk = cite.get("content", cite.get("chunk", cite.get("text", "")))
            print(f"\n  [{i}] {title}")
            if chunk:
                preview = chunk[:200].replace("\n", " ")
                if len(chunk) > 200:
                    preview += "..."
                print(f"      {preview}")

    # Raw response
    print(f"\n{'─' * 60}")
    print("RAW RESPONSE (for debugging)")
    print("─" * 60)
    print(raw[:3000])
    if len(raw) > 3000:
        print("... (truncated)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create and query an Azure AI Search knowledge base (Foundry IQ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name", type=str, default="kb-research-demo-kb",
        help="Knowledge base name (default: kb-research-demo-kb)",
    )
    parser.add_argument(
        "--ks-name", type=str, default="kb-research-demo-ks",
        help="Knowledge source name to reference (default: kb-research-demo-ks)",
    )
    parser.add_argument(
        "--query", type=str,
        default="What are the PPE requirements for working on site?",
        help="Test query to run against the knowledge base",
    )
    parser.add_argument(
        "--skip-create", action="store_true",
        help="Skip creation, only query an existing knowledge base",
    )
    parser.add_argument(
        "--skip-query", action="store_true",
        help="Skip querying, only create the knowledge base",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Azure AI Search — Knowledge Base (Foundry IQ)")
    print("=" * 60)
    print(f"Search endpoint: {get_search_endpoint()}")
    print(f"KB name:         {args.name}")
    print(f"KS name:         {args.ks_name}")
    print()

    # Step 1: List existing knowledge bases
    list_knowledge_bases()

    # Step 2: Create or update the knowledge base
    if not args.skip_create:
        kb = create_or_update_kb(args.name, args.ks_name)
        if kb is None:
            print("\nKnowledge base creation failed. Exiting.")
            sys.exit(1)
    else:
        print("\n(Skipping creation — --skip-create flag set)")

    # Step 3: Run a test query
    if not args.skip_query:
        result = query_knowledge_base(args.name, args.query)
        if result is None:
            print("\nQuery failed.")
            sys.exit(1)
    else:
        print("\n(Skipping query — --skip-query flag set)")

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
