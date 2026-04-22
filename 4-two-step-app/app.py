"""Two-step RAG app — orchestrates retrieve → generate.

Composes retrieve.py and generate.py into a single pipeline with
split timing metrics.  CLI mirrors 1-baseline-app/app.py for drop-in use.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running this file directly
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from retrieve import retrieve, RetrieveResult
from generate import generate, GenerateResult


def get_search_endpoint() -> str:
    val = os.environ.get("AZURE_SEARCH_ENDPOINT", "").strip()
    if not val:
        print("ERROR: AZURE_SEARCH_ENDPOINT not set in .env")
        sys.exit(1)
    return val


def get_openai_endpoint() -> str:
    val = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if not val:
        print("ERROR: AZURE_OPENAI_ENDPOINT not set in .env")
        sys.exit(1)
    return val


def query_pipeline(
    query: str,
    model: str,
    credential: AzureCliCredential,
    search_endpoint: str,
    openai_endpoint: str,
    *,
    cached_retrieve: RetrieveResult | None = None,
) -> dict:
    """Run the full retrieve → generate pipeline.

    If cached_retrieve is provided, skips the retrieve step (used when
    comparing multiple models on the same queries).
    """
    t_start = time.perf_counter()

    # Step 1: Retrieve
    if cached_retrieve:
        ret = cached_retrieve
    else:
        ret = retrieve(query, search_endpoint, credential)

    # Step 2: Generate
    gen = generate(query, ret.context_text, model, openai_endpoint, credential)

    e2e_ms = (time.perf_counter() - t_start) * 1000

    return {
        "query": query,
        "response": gen.response,
        "context": ret.context_text,
        "retrieve_latency_ms": ret.latency_ms,
        "generate_latency_ms": gen.latency_ms,
        "e2e_latency_ms": round(e2e_ms, 1),
        "generate_model": gen.model,
        "prompt_tokens": gen.prompt_tokens,
        "completion_tokens": gen.completion_tokens,
        "total_tokens": gen.total_tokens,
        "num_documents": len(ret.documents),
    }


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def run_interactive(model: str):
    credential = AzureCliCredential()
    search_ep = get_search_endpoint()
    openai_ep = get_openai_endpoint()

    print(f"\n=== Two-Step RAG — {model} ===")
    print(f"Search  : {search_ep}")
    print(f"OpenAI  : {openai_ep}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = query_pipeline(user_input, model, credential, search_ep, openai_ep)
        print(f"\nAssistant: {result['response']}")
        print(f"  [retrieve={result['retrieve_latency_ms']:.0f}ms, "
              f"generate={result['generate_latency_ms']:.0f}ms, "
              f"e2e={result['e2e_latency_ms']:.0f}ms, "
              f"tokens={result['total_tokens']}]\n")


def main():
    parser = argparse.ArgumentParser(description="Two-step RAG app (retrieve → generate)")
    parser.add_argument("--model", default=os.environ.get("FOUNDRY_MODEL_DEPLOYMENT", "gpt-5"), help="Generation model deployment name")
    args = parser.parse_args()
    run_interactive(args.model)


if __name__ == "__main__":
    main()
