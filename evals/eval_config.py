"""Shared configuration for evaluation scripts.

Loads Azure OpenAI settings from .env and exposes model_config dicts
expected by the azure-ai-evaluation SDK evaluators.

This project uses the Azure AI Evaluation SDK for built-in evaluators, but
the eval framework is pluggable — you can swap in any evaluation library
(ragas, deepeval, custom scripts) as long as it reads the same JSONL data files.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from evals/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"ERROR: {name} is not set. Check your .env file.")
        sys.exit(1)
    return val


def get_model_config() -> dict:
    """Return the model_config dict used by AI-assisted evaluators.

    The azure-ai-evaluation SDK expects an AzureOpenAIModelConfiguration.
    For Entra ID auth we acquire a bearer token and pass it as api_key,
    because the SDK's credential validation has a known bug with typing.Any.
    """
    auth_type = os.environ.get("AZURE_OPENAI_AUTH_TYPE", "entra").lower()
    endpoint = _require("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[: -len("/openai")]

    config = {
        "azure_endpoint": endpoint,
        "azure_deployment": _require("AZURE_OPENAI_DEPLOYMENT"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    }

    if auth_type == "entra":
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        config["api_key"] = token.token
    else:
        config["api_key"] = _require("AZURE_OPENAI_API_KEY")

    return config


def get_foundry_project() -> dict | None:
    """Return the azure_ai_project dict for Foundry portal integration, or None.

    When configured, eval results are automatically uploaded to the Azure AI
    Foundry portal where you get dashboards, run comparison, and drill-down.

    This is entirely optional — evals work fine locally without it.
    Set the FOUNDRY_* env vars in .env to enable.
    """
    sub = os.environ.get("FOUNDRY_SUBSCRIPTION_ID", "").strip()
    rg = os.environ.get("FOUNDRY_RESOURCE_GROUP", "").strip()
    project = os.environ.get("FOUNDRY_PROJECT_NAME", "").strip()

    if sub and rg and project:
        return {
            "subscription_id": sub,
            "resource_group_name": rg,
            "project_name": project,
        }
    return None


# Convenience: data directory path
DATA_DIR = Path(__file__).resolve().parent / "data"
