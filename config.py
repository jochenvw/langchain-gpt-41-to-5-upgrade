"""Configuration module for the LangChain Azure OpenAI BYOD application.

Loads environment variables from .env and validates required settings.
Supports both API-key and Entra ID (Azure AD) authentication.
"""

import os
import sys
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AzureOpenAIConfig:
    """Azure OpenAI and Azure AI Search configuration."""

    # Azure OpenAI settings
    endpoint: str
    api_key: str
    api_version: str
    deployment: str
    auth_type: str  # "key" or "entra"

    # Azure AI Search settings (for BYOD / On Your Data)
    search_endpoint: str
    search_index: str
    search_api_key: str
    search_auth_type: str  # "key" or "rbac"

    @property
    def azure_endpoint_base(self) -> str:
        """Return the base endpoint WITHOUT the /openai suffix.

        AzureChatOpenAI's azure_endpoint parameter expects the base URL and
        appends /openai/deployments/... itself.  If the configured endpoint
        already includes /openai we strip it so the SDK builds the right URL.
        """
        base = self.endpoint.rstrip("/")
        if base.endswith("/openai"):
            return base[: -len("/openai")]
        return base

    def get_azure_ad_token_provider(self):
        """Return a callable token provider for Entra ID auth, or None."""
        if self.auth_type != "entra":
            return None
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        credential = DefaultAzureCredential()
        return get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )


def _require_env(name: str) -> str:
    """Return an environment variable's value or exit with an error."""
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        print("       Copy .env.example to .env and fill in the values.")
        sys.exit(1)
    return value


def load_config() -> AzureOpenAIConfig:
    """Load and validate configuration from environment variables."""
    auth_type = os.environ.get("AZURE_OPENAI_AUTH_TYPE", "entra").lower()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if auth_type == "key" and not api_key:
        print("ERROR: AZURE_OPENAI_AUTH_TYPE=key but AZURE_OPENAI_API_KEY is empty.")
        sys.exit(1)

    return AzureOpenAIConfig(
        endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=api_key,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        deployment=_require_env("AZURE_OPENAI_DEPLOYMENT"),
        auth_type=auth_type,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT", ""),
        search_index=os.environ.get("AZURE_SEARCH_INDEX", ""),
        search_api_key=os.environ.get("AZURE_SEARCH_API_KEY", ""),
        search_auth_type=os.environ.get("AZURE_SEARCH_AUTH_TYPE", "rbac").lower(),
    )


# Module-level convenience — import config.settings from other modules
settings = load_config()
