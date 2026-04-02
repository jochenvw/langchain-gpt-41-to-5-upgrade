"""Generate BYOD evaluation dataset from an Azure AI Search index.

Connects directly to a search index, samples documents, uses GPT to
synthesise realistic evaluation queries, then optionally calls the BYOD
pipeline to capture (query, response, context, ground_truth) tuples.

All Azure settings come from .env by default, but can be overridden via CLI
flags — so customers can point this at their own search environment without
editing config files.

Usage:
    # Use defaults from .env
    uv run python scripts/generate_byod_eval_data.py

    # Point at a different search environment
    uv run python scripts/generate_byod_eval_data.py \\
        --search-endpoint https://my-search.search.windows.net \\
        --search-index my-index \\
        --search-auth-type key \\
        --search-api-key <key>

    # Control volume and schema
    uv run python scripts/generate_byod_eval_data.py \\
        --sample-size 20 \\
        --queries-per-doc 3 \\
        --content-fields content,chunk,text

    # Preview what's in the index without generating anything
    uv run python scripts/generate_byod_eval_data.py --dry-run
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_PATH = PROJECT_ROOT / "evals" / "data" / "byod_test_data.jsonl"

# ---------------------------------------------------------------------------
# Azure AI Search — sample documents
# ---------------------------------------------------------------------------

def get_search_client(
    endpoint: str | None = None,
    index: str | None = None,
    auth_type: str | None = None,
    api_key: str | None = None,
):
    """Build an Azure AI Search client.

    CLI flags take precedence over .env values, so customers can point at
    their own search environment without modifying config files.
    """
    from azure.search.documents import SearchClient

    endpoint = (endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT", "")).strip()
    index = (index or os.environ.get("AZURE_SEARCH_INDEX", "")).strip()
    auth_type = (auth_type or os.environ.get("AZURE_SEARCH_AUTH_TYPE", "token")).lower()
    api_key = (api_key or os.environ.get("AZURE_SEARCH_API_KEY", "")).strip()

    if not endpoint or not index:
        print("ERROR: AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX must be set.")
        sys.exit(1)

    if auth_type == "key":
        from azure.core.credentials import AzureKeyCredential
        if not api_key:
            print("ERROR: --search-api-key (or AZURE_SEARCH_API_KEY) must be set for key auth.")
            sys.exit(1)
        credential = AzureKeyCredential(api_key)
    else:
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()

    return SearchClient(endpoint=endpoint, index_name=index, credential=credential)


def sample_documents(client, sample_size: int, content_fields: list[str]) -> list[dict]:
    """Pull a sample of documents from the search index.

    Uses a wildcard search and random sampling to get a representative set.
    `content_fields` controls which index fields to check for document text
    (varies across search indexes).
    """
    print(f"Querying index for up to {sample_size * 3} candidates to sample from...")
    print(f"Content fields to try: {', '.join(content_fields)}")

    results = client.search(
        search_text="*",
        top=min(sample_size * 3, 1000),
        include_total_count=True,
    )

    docs = []
    for result in results:
        doc = dict(result)
        # Try each content field in order
        content = ""
        for field in content_fields:
            content = doc.get(field, "")
            if content and len(content.strip()) >= 50:
                break
        if not content or len(content.strip()) < 50:
            continue
        docs.append({
            "id": doc.get("id", doc.get("chunk_id", doc.get("uid", ""))),
            "title": (
                doc.get("title")
                or doc.get("metadata_storage_name")
                or doc.get("blob_url", "").rsplit("/", 1)[-1]
                or doc.get("uid", "")
            ),
            "content": content.strip(),
        })

    if not docs:
        print("ERROR: No documents with usable content found in the index.")
        print("       Check the index schema — expected a 'content', 'chunk', or 'text' field.")
        sys.exit(1)

    total = results.get_count()
    print(f"Found {total} total documents, {len(docs)} with usable content.")

    if len(docs) > sample_size:
        docs = random.sample(docs, sample_size)
        print(f"Randomly sampled {sample_size} documents.")
    else:
        print(f"Using all {len(docs)} documents (fewer than requested sample size).")

    return docs


# ---------------------------------------------------------------------------
# Query generation — use GPT to create realistic eval questions
# ---------------------------------------------------------------------------

def generate_queries(docs: list[dict], queries_per_doc: int) -> list[dict]:
    """Use GPT to synthesise evaluation queries from document content."""
    from openai import AzureOpenAI
    from config import settings

    print(f"\nGenerating {queries_per_doc} query(s) per document using {settings.deployment}...")

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

    system_prompt = f"""You are an evaluation dataset generator. Given a document excerpt,
generate exactly {queries_per_doc} realistic question(s) that a user would ask and that
this document can answer. Also provide a concise ground-truth answer for each question
based ONLY on the document content.

Return a JSON array of objects with "query" and "ground_truth" keys. No markdown fencing.

Example output:
[{{"query": "What PPE is required for welding?", "ground_truth": "Welding requires a face shield, heat-resistant gloves, and a leather apron."}}]"""

    eval_items = []
    for i, doc in enumerate(docs):
        # Truncate very long documents to fit context
        excerpt = doc["content"][:3000]
        title_hint = f" (from: {doc['title']})" if doc["title"] else ""

        try:
            response = client.chat.completions.create(
                model=settings.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document{title_hint}:\n\n{excerpt}"},
                ],
                temperature=0.7,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fencing if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[: raw.rfind("```")]
            items = json.loads(raw)

            for item in items[:queries_per_doc]:
                eval_items.append({
                    "query": item["query"],
                    "ground_truth": item.get("ground_truth", ""),
                    "source_doc_id": doc["id"],
                    "source_doc_title": doc["title"],
                })

            print(f"  [{i+1}/{len(docs)}] {doc['title'][:50]:<50} → {len(items)} queries")

        except Exception as exc:
            print(f"  [{i+1}/{len(docs)}] FAILED: {exc}")
            continue

        # Brief pause to respect rate limits
        time.sleep(0.5)

    print(f"\nGenerated {len(eval_items)} total eval queries.")
    return eval_items


# ---------------------------------------------------------------------------
# BYOD pipeline — get responses and context for each query
# ---------------------------------------------------------------------------

def run_byod_pipeline(eval_items: list[dict]) -> list[dict]:
    """Run each query through the BYOD pipeline to capture response + context."""
    from app import build_llm, get_byod_extra_body
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\nRunning {len(eval_items)} queries through BYOD pipeline...")

    llm = build_llm()
    extra_body = get_byod_extra_body()
    system = SystemMessage(content="You are a helpful assistant.")

    for i, item in enumerate(eval_items):
        try:
            messages = [system, HumanMessage(content=item["query"])]
            result = llm.invoke(messages, extra_body=extra_body)

            item["response"] = result.content

            # Extract context from citations
            context = ""
            if hasattr(result, "response_metadata") and result.response_metadata:
                meta = result.response_metadata
                ctx_block = meta.get("context", {})
                citations = ctx_block.get("citations") or ctx_block.get("documents") or []
                context = "\n\n".join(
                    c.get("content", "") for c in citations if c.get("content")
                )
            item["context"] = context

            status = f"{len(item['response'])} chars"
            print(f"  [{i+1}/{len(eval_items)}] {item['query'][:55]:<55} → {status}")

        except Exception as exc:
            print(f"  [{i+1}/{len(eval_items)}] FAILED: {exc}")
            item["response"] = ""
            item["context"] = ""

        time.sleep(0.3)

    successful = sum(1 for it in eval_items if it["response"])
    print(f"\nBYOD pipeline complete: {successful}/{len(eval_items)} succeeded.")
    return eval_items


# ---------------------------------------------------------------------------
# Write JSONL output
# ---------------------------------------------------------------------------

def write_jsonl(eval_items: list[dict], output_path: Path) -> None:
    """Write eval dataset as JSONL in the format expected by eval_byod.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in eval_items:
            row = {
                "query": item["query"],
                "response": item.get("response", ""),
                "context": item.get("context", ""),
                "ground_truth": item.get("ground_truth", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(eval_items)} rows to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CONTENT_FIELDS = ["content", "snippet", "chunk", "text", "merged_content", "description"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate BYOD eval dataset from Azure AI Search index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (from .env or shell):
  AZURE_SEARCH_ENDPOINT     Search service URL
  AZURE_SEARCH_INDEX        Index name
  AZURE_SEARCH_AUTH_TYPE    token | key | rbac  (default: token)
  AZURE_SEARCH_API_KEY      Required when auth type is 'key'

CLI flags override .env values, so you can point at any search environment:
  uv run python scripts/generate_byod_eval_data.py \\
      --search-endpoint https://other.search.windows.net \\
      --search-index other-index --search-auth-type key --search-api-key <key>
""",
    )

    # --- Search environment overrides ---
    search_group = parser.add_argument_group("Search environment (override .env)")
    search_group.add_argument(
        "--search-endpoint", type=str, default=None,
        help="Azure AI Search endpoint URL (overrides AZURE_SEARCH_ENDPOINT)",
    )
    search_group.add_argument(
        "--search-index", type=str, default=None,
        help="Search index name (overrides AZURE_SEARCH_INDEX)",
    )
    search_group.add_argument(
        "--search-auth-type", type=str, default=None,
        choices=["token", "key", "rbac"],
        help="Search auth method (overrides AZURE_SEARCH_AUTH_TYPE)",
    )
    search_group.add_argument(
        "--search-api-key", type=str, default=None,
        help="Search API key — required when --search-auth-type=key",
    )

    # --- Generation options ---
    gen_group = parser.add_argument_group("Generation options")
    gen_group.add_argument(
        "--sample-size", type=int, default=10,
        help="Number of documents to sample from the index (default: 10)",
    )
    gen_group.add_argument(
        "--queries-per-doc", type=int, default=2,
        help="Number of eval queries to generate per document (default: 2)",
    )
    gen_group.add_argument(
        "--content-fields", type=str, default=",".join(DEFAULT_CONTENT_FIELDS),
        help=(
            "Comma-separated list of index fields to check for document text, "
            f"in priority order (default: {','.join(DEFAULT_CONTENT_FIELDS)})"
        ),
    )
    gen_group.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help=f"Output JSONL path (default: {OUTPUT_PATH})",
    )
    gen_group.add_argument(
        "--skip-byod", action="store_true",
        help="Skip the BYOD pipeline step (only generate queries, no responses)",
    )
    gen_group.add_argument(
        "--dry-run", action="store_true",
        help="Sample docs and show what would be generated, but don't call GPT",
    )
    args = parser.parse_args()

    output = Path(args.output)
    content_fields = [f.strip() for f in args.content_fields.split(",") if f.strip()]

    # Resolve effective search config (CLI > .env)
    effective_endpoint = args.search_endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    effective_index = args.search_index or os.environ.get("AZURE_SEARCH_INDEX", "")

    print("=" * 60)
    print("BYOD Eval Dataset Generator")
    print("=" * 60)
    print(f"Search endpoint : {effective_endpoint}")
    print(f"Search index    : {effective_index}")
    print(f"Content fields  : {', '.join(content_fields)}")
    print(f"Sample size     : {args.sample_size} documents")
    print(f"Queries per doc : {args.queries_per_doc}")
    print(f"Output          : {output}")
    print()

    # Step 1: Sample documents from search index
    search_client = get_search_client(
        endpoint=args.search_endpoint,
        index=args.search_index,
        auth_type=args.search_auth_type,
        api_key=args.search_api_key,
    )
    docs = sample_documents(search_client, args.sample_size, content_fields)

    if args.dry_run:
        print("\n--- DRY RUN — sampled documents ---")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc['title'][:60]} ({len(doc['content'])} chars)")
        print(f"\nWould generate ~{len(docs) * args.queries_per_doc} queries. Exiting.")
        return

    # Step 2: Generate queries from document content
    eval_items = generate_queries(docs, args.queries_per_doc)
    if not eval_items:
        print("ERROR: No queries generated. Check GPT connectivity.")
        sys.exit(1)

    # Step 3: Run through BYOD pipeline (optional)
    if not args.skip_byod:
        eval_items = run_byod_pipeline(eval_items)

    # Step 4: Write output
    write_jsonl(eval_items, output)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(eval_items)} eval queries ready.")
    print(f"Run evals with: uv run python -m evals.run_all --suite byod")


if __name__ == "__main__":
    main()
