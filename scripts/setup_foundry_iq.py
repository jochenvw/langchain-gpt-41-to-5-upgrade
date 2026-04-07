"""Provision Foundry IQ resources: knowledge source, knowledge base, and project connection.

This script sets up the Azure AI Search agentic retrieval pipeline and connects
it to the Azure AI Foundry project so agents can use Foundry IQ for grounding.

Prerequisites:
  - Azure AI Search with semantic ranker enabled
  - Azure AI Foundry project with a GPT deployment
  - RBAC: Search Service Contributor + Search Index Data Reader on the search service
  - RBAC: Azure AI Project Manager on the Foundry project

Usage:
    python scripts/setup_foundry_iq.py              # create everything
    python scripts/setup_foundry_iq.py --check       # check existing resources
    python scripts/setup_foundry_iq.py --teardown    # delete created resources

Required .env variables:
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX
    FOUNDRY_ENDPOINT (project endpoint)
    FOUNDRY_MODEL_DEPLOYMENT (e.g. gpt-5)
    AZURE_OPENAI_ENDPOINT (for knowledge base LLM connection)

Optional .env variables:
    FOUNDRY_KNOWLEDGE_BASE_NAME  (default: safety-knowledge-base)
    FOUNDRY_MCP_CONNECTION_NAME  (default: safety-kb-mcp)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "")
AOAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
if AOAI_ENDPOINT.endswith("/openai"):
    AOAI_ENDPOINT = AOAI_ENDPOINT[: -len("/openai")]
MODEL_DEPLOYMENT = os.environ.get("FOUNDRY_MODEL_DEPLOYMENT", "gpt-5")
FOUNDRY_ENDPOINT = os.environ.get("FOUNDRY_ENDPOINT", "")

KB_NAME = os.environ.get("FOUNDRY_KNOWLEDGE_BASE_NAME", "safety-knowledge-base")
KS_NAME = f"{KB_NAME}-source"
MCP_CONN_NAME = os.environ.get("FOUNDRY_MCP_CONNECTION_NAME", "safety-kb-mcp")

API_VERSION = "2025-11-01-preview"

credential = DefaultAzureCredential()


def _search_headers() -> dict:
    """Get auth headers for Azure AI Search REST calls."""
    token = credential.get_token("https://search.azure.com/.default")
    return {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json",
    }


def _arm_headers() -> dict:
    """Get auth headers for Azure Resource Manager REST calls."""
    provider = get_bearer_token_provider(
        credential, "https://management.azure.com/.default"
    )
    return {
        "Authorization": f"Bearer {provider()}",
        "Content-Type": "application/json",
    }


def _project_resource_id() -> str:
    """Derive the ARM resource ID from the Foundry project endpoint.

    Endpoint format: https://<account>.services.ai.azure.com/api/projects/<project>
    We also need subscription and resource group, which we can extract from
    FOUNDRY_SEARCH_CONNECTION_ID or from explicit env vars.
    """
    rid = os.environ.get("FOUNDRY_PROJECT_RESOURCE_ID", "").strip()
    if rid:
        return rid

    # Extract account and project from the endpoint
    import re
    m = re.match(
        r"https://([^.]+)\.services\.ai\.azure\.com/api/projects/([^/]+)",
        FOUNDRY_ENDPOINT,
    )
    if not m:
        _missing_rid_error()
    account, project = m.group(1), m.group(2)

    # Try explicit env vars first
    sub = os.environ.get("FOUNDRY_SUBSCRIPTION_ID", "").strip()
    rg = os.environ.get("FOUNDRY_RESOURCE_GROUP", "").strip()

    # Fall back: extract from FOUNDRY_SEARCH_CONNECTION_ID
    if not (sub and rg):
        conn_id = os.environ.get("FOUNDRY_SEARCH_CONNECTION_ID", "")
        m2 = re.match(
            r"/subscriptions/([^/]+)/resourceGroups/([^/]+)/", conn_id
        )
        if m2:
            sub = sub or m2.group(1)
            rg = rg or m2.group(2)

    if sub and rg:
        return (
            f"/subscriptions/{sub}/resourceGroups/{rg}"
            f"/providers/Microsoft.CognitiveServices"
            f"/accounts/{account}/projects/{project}"
        )

    _missing_rid_error()


def _missing_rid_error():
    print("ERROR: Cannot determine project ARM resource ID.")
    print("       Set FOUNDRY_PROJECT_RESOURCE_ID in .env, or set")
    print("       FOUNDRY_SUBSCRIPTION_ID + FOUNDRY_RESOURCE_GROUP")
    sys.exit(1)


# ── Knowledge Source ────────────────────────────────────────────────


def check_knowledge_source() -> bool:
    url = f"{SEARCH_ENDPOINT}/knowledgesources/{KS_NAME}"
    r = requests.get(url, params={"api-version": API_VERSION}, headers=_search_headers())
    return r.status_code == 200


def create_knowledge_source():
    print(f"Creating knowledge source '{KS_NAME}' → index '{SEARCH_INDEX}'...")
    url = f"{SEARCH_ENDPOINT}/knowledgesources/{KS_NAME}"
    body = {
        "name": KS_NAME,
        "kind": "searchIndex",
        "description": f"Knowledge source backed by the {SEARCH_INDEX} index.",
        "searchIndexParameters": {
            "searchIndexName": SEARCH_INDEX,
            "semanticConfigurationName": "safety-source-semantic-configuration",
            "sourceDataFields": [
                {"name": "uid"},
                {"name": "snippet"},
                {"name": "blob_url"},
            ],
        },
    }
    r = requests.put(
        url,
        params={"api-version": API_VERSION},
        headers=_search_headers(),
        json=body,
    )
    r.raise_for_status()
    print(f"  ✓ Knowledge source '{KS_NAME}' created.")


def delete_knowledge_source():
    url = f"{SEARCH_ENDPOINT}/knowledgesources/{KS_NAME}"
    r = requests.delete(url, params={"api-version": API_VERSION}, headers=_search_headers())
    if r.status_code in (200, 204, 404):
        print(f"  ✓ Knowledge source '{KS_NAME}' deleted.")
    else:
        print(f"  ✗ Failed to delete knowledge source: {r.status_code} {r.text}")


# ── Knowledge Base ──────────────────────────────────────────────────


def check_knowledge_base() -> bool:
    url = f"{SEARCH_ENDPOINT}/knowledgebases/{KB_NAME}"
    r = requests.get(url, params={"api-version": API_VERSION}, headers=_search_headers())
    return r.status_code == 200


def create_knowledge_base():
    print(f"Creating knowledge base '{KB_NAME}'...")
    url = f"{SEARCH_ENDPOINT}/knowledgebases/{KB_NAME}"
    body = {
        "name": KB_NAME,
        "description": "Safety compliance knowledge base for agentic retrieval.",
        "knowledgeSources": [{"name": KS_NAME}],
        "models": [
            {
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                    "resourceUri": AOAI_ENDPOINT,
                    "deploymentId": MODEL_DEPLOYMENT,
                    "modelName": MODEL_DEPLOYMENT,
                },
            }
        ],
        "retrievalReasoningEffort": {"kind": "low"},
        "outputMode": "answerSynthesis",
        "retrievalInstructions": (
            "Use the safety-source knowledge to answer questions about "
            "safety compliance, regulations, incident reporting, and risk management."
        ),
    }
    r = requests.put(
        url,
        params={"api-version": API_VERSION},
        headers=_search_headers(),
        json=body,
    )
    r.raise_for_status()
    print(f"  ✓ Knowledge base '{KB_NAME}' created.")


def delete_knowledge_base():
    url = f"{SEARCH_ENDPOINT}/knowledgebases/{KB_NAME}"
    r = requests.delete(url, params={"api-version": API_VERSION}, headers=_search_headers())
    if r.status_code in (200, 204, 404):
        print(f"  ✓ Knowledge base '{KB_NAME}' deleted.")
    else:
        print(f"  ✗ Failed to delete knowledge base: {r.status_code} {r.text}")


# ── Project Connection (MCP) ───────────────────────────────────────


def mcp_endpoint() -> str:
    return (
        f"{SEARCH_ENDPOINT}/knowledgebases/{KB_NAME}"
        f"/mcp?api-version={API_VERSION}"
    )


def check_project_connection() -> bool:
    rid = _project_resource_id()
    url = (
        f"https://management.azure.com{rid}"
        f"/connections/{MCP_CONN_NAME}"
        f"?api-version=2025-10-01-preview"
    )
    r = requests.get(url, headers=_arm_headers())
    return r.status_code == 200


def create_project_connection():
    print(f"Creating project connection '{MCP_CONN_NAME}'...")
    rid = _project_resource_id()
    url = (
        f"https://management.azure.com{rid}"
        f"/connections/{MCP_CONN_NAME}"
        f"?api-version=2025-10-01-preview"
    )
    body = {
        "name": MCP_CONN_NAME,
        "type": "Microsoft.MachineLearningServices/workspaces/connections",
        "properties": {
            "authType": "ProjectManagedIdentity",
            "category": "RemoteTool",
            "target": mcp_endpoint(),
            "isSharedToAll": True,
            "audience": "https://search.azure.com/",
            "metadata": {"ApiType": "Azure"},
        },
    }
    r = requests.put(url, headers=_arm_headers(), json=body)
    r.raise_for_status()
    print(f"  ✓ Project connection '{MCP_CONN_NAME}' created.")
    print(f"    MCP endpoint: {mcp_endpoint()}")


def delete_project_connection():
    rid = _project_resource_id()
    url = (
        f"https://management.azure.com{rid}"
        f"/connections/{MCP_CONN_NAME}"
        f"?api-version=2025-10-01-preview"
    )
    r = requests.delete(url, headers=_arm_headers())
    if r.status_code in (200, 204, 404):
        print(f"  ✓ Project connection '{MCP_CONN_NAME}' deleted.")
    else:
        print(f"  ✗ Failed to delete connection: {r.status_code} {r.text}")


# ── Main ────────────────────────────────────────────────────────────


def check():
    print("Checking Foundry IQ resources...")
    print(f"  Knowledge source '{KS_NAME}': {'exists' if check_knowledge_source() else 'NOT FOUND'}")
    print(f"  Knowledge base   '{KB_NAME}': {'exists' if check_knowledge_base() else 'NOT FOUND'}")
    try:
        conn = check_project_connection()
        print(f"  Project connection '{MCP_CONN_NAME}': {'exists' if conn else 'NOT FOUND'}")
    except Exception as e:
        print(f"  Project connection '{MCP_CONN_NAME}': error checking ({e})")


def setup():
    print("=" * 60)
    print("Foundry IQ Setup — Knowledge Base + MCP Connection")
    print("=" * 60)
    print(f"Search endpoint : {SEARCH_ENDPOINT}")
    print(f"Search index    : {SEARCH_INDEX}")
    print(f"KB name         : {KB_NAME}")
    print(f"MCP connection  : {MCP_CONN_NAME}")
    print(f"Model           : {MODEL_DEPLOYMENT}")
    print()

    if not check_knowledge_source():
        create_knowledge_source()
    else:
        print(f"  ✓ Knowledge source '{KS_NAME}' already exists.")

    if not check_knowledge_base():
        create_knowledge_base()
    else:
        print(f"  ✓ Knowledge base '{KB_NAME}' already exists.")

    try:
        if not check_project_connection():
            create_project_connection()
        else:
            print(f"  ✓ Project connection '{MCP_CONN_NAME}' already exists.")
    except Exception as e:
        print(f"  ⚠ Could not check/create project connection: {e}")
        print(f"    You may need to set FOUNDRY_PROJECT_RESOURCE_ID in .env")

    print()
    print("Add the following to .env if not already set:")
    print(f"  FOUNDRY_KNOWLEDGE_BASE_NAME={KB_NAME}")
    print(f"  FOUNDRY_MCP_CONNECTION_NAME={MCP_CONN_NAME}")
    print()
    print("REQUIRED RBAC (needs Owner / User Access Administrator):")
    print("  The project's managed identity needs 'Search Index Data Reader'")
    print("  role on the Azure AI Search service. Run:")
    print()
    print("  az role assignment create \\")
    print(f"    --assignee <project-managed-identity-principal-id> \\")
    print(f"    --role 'Search Index Data Reader' \\")
    print(f"    --scope <search-service-arm-resource-id>")
    print()
    print("Done! Run `python app.py --mode foundry` to test.")


def teardown():
    print("Tearing down Foundry IQ resources...")
    try:
        delete_project_connection()
    except Exception as e:
        print(f"  ⚠ Connection deletion failed: {e}")
    delete_knowledge_base()
    delete_knowledge_source()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Provision Foundry IQ resources")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Check existing resources")
    group.add_argument("--teardown", action="store_true", help="Delete resources")
    args = parser.parse_args()

    for var in ("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_INDEX", "FOUNDRY_ENDPOINT"):
        if not os.environ.get(var, "").strip():
            print(f"ERROR: {var} is not set. Check your .env file.")
            sys.exit(1)

    if args.check:
        check()
    elif args.teardown:
        teardown()
    else:
        setup()


if __name__ == "__main__":
    main()
