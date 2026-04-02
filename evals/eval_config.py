"""Shared configuration for evaluation scripts.

Loads Azure OpenAI settings from .env and exposes model_config dicts
expected by the azure-ai-evaluation SDK evaluators.
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

    Uses Entra ID (token-based) auth when AZURE_OPENAI_AUTH_TYPE=entra,
    otherwise falls back to API key.
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
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        credential = DefaultAzureCredential()
        config["azure_ad_token_provider"] = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
    else:
        config["api_key"] = _require("AZURE_OPENAI_API_KEY")

    return config


# Convenience: data directory path
DATA_DIR = Path(__file__).resolve().parent / "data"
