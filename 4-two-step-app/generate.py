"""Stage 2 of the two-step RAG pipeline: Answer generation.

Calls the Azure OpenAI chat completions API with retrieved context
injected into the system prompt.  Returns the answer and timing/token
metadata.
"""

import time
from dataclasses import dataclass

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful safety assistant. Answer the user's question using "
    "ONLY the provided context. If the context does not contain enough "
    "information, say so. Cite the source document titles when possible."
)
MAX_COMPLETION_TOKENS = 1024


@dataclass
class GenerateResult:
    """Normalised output of a generation call."""
    response: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _build_client(
    endpoint: str,
    credential: AzureCliCredential,
    api_version: str = "2024-06-01",
) -> AzureOpenAI:
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    base = endpoint.rstrip("/")
    if base.endswith("/openai"):
        base = base[: -len("/openai")]

    return AzureOpenAI(
        azure_endpoint=base,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )


def generate(
    query: str,
    context: str,
    model: str,
    endpoint: str,
    credential: AzureCliCredential,
    *,
    api_version: str = "2024-06-01",
    max_tokens: int = MAX_COMPLETION_TOKENS,
    temperature: float = 0.3,
    retries: int = 2,
) -> GenerateResult:
    """Generate an answer using the Azure OpenAI chat completions API.

    Retries on transient failures (429, 500, timeout).
    """
    client = _build_client(endpoint, credential, api_version)

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n## Context\n{context}"},
        {"role": "user", "content": query},
    ]

    last_exc = None
    for attempt in range(1 + retries):
        try:
            t0 = time.perf_counter()
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            choice = completion.choices[0] if completion.choices else None
            text = (choice.message.content or "") if choice else ""

            usage = completion.usage
            return GenerateResult(
                response=text,
                model=completion.model or model,
                latency_ms=round(latency_ms, 1),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            if attempt < retries and any(k in err for k in ("429", "500", "timeout", "rate")):
                import time as _time
                _time.sleep(2 ** attempt)
                continue
            raise

    raise last_exc  # type: ignore[misc]
