"""Create a Knowledge Source in Azure AI Search (2025-11-01-preview).

A knowledge source is an abstraction layer introduced in the agentic retrieval
preview API.  It wraps an existing search index and tells the agentic retrieval
pipeline *which* index to query and *how* to interpret the results (semantic
config, citation fields, etc.).

This script attempts the SDK-first approach using ``SearchIndexKnowledgeSource``
from ``azure-search-documents``.  If those classes aren't available in the
installed package version it falls back to the REST API with ``requests``.

Usage
-----
    # Token auth (default) — uses DefaultAzureCredential
    python 02_create_knowledge_source.py

    # Override names
    python 02_create_knowledge_source.py --name my-ks --index-name my-idx

    # API-key auth
    python 02_create_knowledge_source.py --auth-type key
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

API_VERSION = "2025-11-01-preview"

DEFAULT_KS_NAME = "kb-research-demo-ks"
DEFAULT_INDEX_NAME = "kb-research-demo"
DEFAULT_SEMANTIC_CONFIG = "my-semantic-config"

# Fields surfaced in citations by the agentic retrieval pipeline.
SOURCE_DATA_FIELDS = ["title", "source_url", "category"]

# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a knowledge source in Azure AI Search.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_KS_NAME,
        help=f"Knowledge source name (default: {DEFAULT_KS_NAME})",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help=f"Target search index name (default: {DEFAULT_INDEX_NAME})",
    )
    parser.add_argument(
        "--semantic-config",
        default=DEFAULT_SEMANTIC_CONFIG,
        help=f"Semantic configuration name (default: {DEFAULT_SEMANTIC_CONFIG})",
    )
    parser.add_argument(
        "--auth-type",
        choices=["token", "key"],
        default=None,
        help="Auth method. Falls back to AZURE_SEARCH_AUTH_TYPE env var, then 'token'.",
    )
    return parser.parse_args()


def _resolve_auth_type(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return os.environ.get("AZURE_SEARCH_AUTH_TYPE", "token").lower()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        print("       Copy .env.example to .env and fill in the values.")
        sys.exit(1)
    return value


# ── SDK approach ─────────────────────────────────────────────────────────────


def _try_sdk(
    endpoint: str,
    auth_type: str,
    ks_name: str,
    index_name: str,
    semantic_config: str,
) -> bool:
    """Try to create the knowledge source via the Python SDK.

    Returns True if the SDK path succeeded, False if the required classes are
    not available (caller should fall back to REST).
    """
    try:
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchIndexFieldReference,
            SearchIndexKnowledgeSource,
            SearchIndexKnowledgeSourceParameters,
        )
    except ImportError:
        return False

    print("✓ SDK classes available — using Python SDK path.\n")

    # Build client
    if auth_type == "key":
        from azure.core.credentials import AzureKeyCredential

        api_key = _require_env("AZURE_SEARCH_API_KEY")
        credential = AzureKeyCredential(api_key)
    else:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()

    index_client = SearchIndexClient(
        endpoint=endpoint,
        credential=credential,
        api_version=API_VERSION,
    )

    # ── List existing knowledge sources ──────────────────────────────────
    print("1 · Listing existing knowledge sources …")
    try:
        existing = list(index_client.list_knowledge_sources())
        if existing:
            for ks in existing:
                print(f"   • {ks.name}")
        else:
            print("   (none found)")
    except Exception as exc:
        print(f"   ⚠  Could not list knowledge sources: {exc}")
    print()

    # ── Build the knowledge source definition ────────────────────────────
    print("2 · Building knowledge source definition …")
    print(f"   Name               : {ks_name}")
    print(f"   Target index       : {index_name}")
    print(f"   Semantic config    : {semantic_config}")
    print(f"   Source data fields : {', '.join(SOURCE_DATA_FIELDS)}")
    print()

    knowledge_source = SearchIndexKnowledgeSource(
        name=ks_name,
        description=f"Knowledge source for the {index_name} index.",
        search_index_parameters=SearchIndexKnowledgeSourceParameters(
            search_index_name=index_name,
            semantic_configuration_name=semantic_config,
            source_data_fields=[
                SearchIndexFieldReference(name=field)
                for field in SOURCE_DATA_FIELDS
            ],
            search_fields=[],  # empty = search all searchable fields
        ),
    )

    # ── Create or update ─────────────────────────────────────────────────
    print("3 · Creating / updating knowledge source …")
    result = index_client.create_or_update_knowledge_source(knowledge_source)
    print(f"   ✓ Knowledge source '{result.name}' created/updated.\n")

    # ── Verify by retrieving ─────────────────────────────────────────────
    print("4 · Verifying — retrieving knowledge source definition …")
    retrieved = index_client.get_knowledge_source(ks_name)
    print(f"   Name           : {retrieved.name}")
    print(f"   Description    : {retrieved.description}")
    params = retrieved.search_index_parameters
    if params:
        print(f"   Index          : {params.search_index_name}")
        print(f"   Semantic cfg   : {params.semantic_configuration_name}")
        if params.source_data_fields:
            fields = ", ".join(f.name for f in params.source_data_fields)
            print(f"   Data fields    : {fields}")
    print()
    return True


# ── REST fallback ────────────────────────────────────────────────────────────


def _rest_fallback(
    endpoint: str,
    auth_type: str,
    ks_name: str,
    index_name: str,
    semantic_config: str,
) -> None:
    """Create the knowledge source using the REST API directly."""
    import requests

    print("⚠  SDK classes not available — falling back to REST API.\n")

    endpoint = endpoint.rstrip("/")

    # ── Auth ─────────────────────────────────────────────────────────────
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if auth_type == "key":
        api_key = _require_env("AZURE_SEARCH_API_KEY")
        headers["api-key"] = api_key
    else:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        token = credential.get_token("https://search.azure.com/.default")
        headers["Authorization"] = f"Bearer {token.token}"

    params = {"api-version": API_VERSION}

    # ── List existing knowledge sources ──────────────────────────────────
    print("1 · Listing existing knowledge sources …")
    list_url = f"{endpoint}/knowledgesources"
    try:
        resp = requests.get(list_url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        existing = resp.json().get("value", [])
        if existing:
            for ks in existing:
                print(f"   • {ks['name']}")
        else:
            print("   (none found)")
    except Exception as exc:
        print(f"   ⚠  Could not list knowledge sources: {exc}")
    print()

    # ── Build the knowledge source body ──────────────────────────────────
    print("2 · Building knowledge source definition …")
    print(f"   Name               : {ks_name}")
    print(f"   Target index       : {index_name}")
    print(f"   Semantic config    : {semantic_config}")
    print(f"   Source data fields : {', '.join(SOURCE_DATA_FIELDS)}")
    print()

    body = {
        "name": ks_name,
        "kind": "searchIndex",
        "description": f"Knowledge source for the {index_name} index.",
        "searchIndexParameters": {
            "searchIndexName": index_name,
            "semanticConfigurationName": semantic_config,
            "sourceDataFields": [{"name": f} for f in SOURCE_DATA_FIELDS],
            "searchFields": [],
        },
    }

    # ── Create or update (PUT) ───────────────────────────────────────────
    print("3 · Creating / updating knowledge source …")
    put_url = f"{endpoint}/knowledgesources('{ks_name}')"
    resp = requests.put(
        put_url, headers=headers, params=params, json=body, timeout=30,
    )
    if not resp.ok:
        print(f"   ✗ PUT failed ({resp.status_code}):")
        print(f"     {resp.text}")
        sys.exit(1)
    print(f"   ✓ Knowledge source '{ks_name}' created/updated.\n")

    # ── Verify by retrieving ─────────────────────────────────────────────
    print("4 · Verifying — retrieving knowledge source definition …")
    get_url = f"{endpoint}/knowledgesources('{ks_name}')"
    resp = requests.get(get_url, headers=headers, params=params, timeout=30)
    if resp.ok:
        ks_def = resp.json()
        print(json.dumps(ks_def, indent=2))
    else:
        print(f"   ⚠  Could not retrieve knowledge source ({resp.status_code}).")
        print(f"     {resp.text}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _get_args()
    auth_type = _resolve_auth_type(args.auth_type)
    endpoint = _require_env("AZURE_SEARCH_ENDPOINT")

    print("=" * 65)
    print("  Azure AI Search — Create Knowledge Source")
    print("  API version:", API_VERSION)
    print("=" * 65)
    print()
    print(f"  Search endpoint : {endpoint}")
    print(f"  Auth type       : {auth_type}")
    print()

    sdk_ok = _try_sdk(
        endpoint=endpoint,
        auth_type=auth_type,
        ks_name=args.name,
        index_name=args.index_name,
        semantic_config=args.semantic_config,
    )

    if not sdk_ok:
        _rest_fallback(
            endpoint=endpoint,
            auth_type=auth_type,
            ks_name=args.name,
            index_name=args.index_name,
            semantic_config=args.semantic_config,
        )

    print("Done ✓")


if __name__ == "__main__":
    main()
