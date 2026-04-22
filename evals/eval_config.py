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


def get_model_config(deployment_override: str | None = None) -> dict:
    """Return the model_config dict for the TARGET model (the one being evaluated).

    Args:
        deployment_override: If set, use this deployment name instead of .env.

    The azure-ai-evaluation SDK expects an AzureOpenAIModelConfiguration.
    For Entra ID auth we acquire a bearer token and pass it as api_key,
    because the SDK's credential validation has a known bug with typing.Any.
    """
    auth_type = os.environ.get("AZURE_OPENAI_AUTH_TYPE", "entra").lower()
    endpoint = _require("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[: -len("/openai")]

    deployment = deployment_override or _require("AZURE_OPENAI_DEPLOYMENT")

    config = {
        "azure_endpoint": endpoint,
        "azure_deployment": deployment,
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


def get_judge_model_config(deployment_override: str | None = None) -> dict:
    """Return the model_config dict for the JUDGE model (the one scoring eval results).

    Args:
        deployment_override: If set, use this deployment name instead of .env.

    Uses AZURE_EVAL_DEPLOYMENT if set, otherwise falls back to the target model.
    Best practice: use a stronger model as judge (e.g. gpt-5.4) than the target.
    """
    judge_deployment = deployment_override or os.environ.get("AZURE_EVAL_DEPLOYMENT", "").strip()
    if not judge_deployment:
        return get_model_config()

    auth_type = os.environ.get("AZURE_OPENAI_AUTH_TYPE", "entra").lower()
    endpoint = os.environ.get("AZURE_EVAL_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        endpoint = _require("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[: -len("/openai")]

    config = {
        "azure_endpoint": endpoint,
        "azure_deployment": judge_deployment,
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
