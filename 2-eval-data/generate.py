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

from shared.prompts import load as load_prompt

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


def normalize_doc_id(doc: dict) -> str:
    """Extract a document ID using a consistent fallback order.

    Azure AI Search indexes use different ID field names depending on
    the indexing pipeline.  This helper ensures we compare IDs
    consistently across sampling, query generation, and retrieval
    validation.
    """
    return str(doc.get("id", doc.get("chunk_id", doc.get("uid", "")))).strip()


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
            "id": normalize_doc_id(doc),
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

def generate_queries(docs: list[dict], queries_per_doc: int, deployment: str | None = None) -> list[dict]:
    """Use GPT to synthesise evaluation queries from document content.

    Args:
        deployment: If set, use this model deployment instead of .env default.
    """
    from openai import AzureOpenAI
    from config import settings

    model = deployment or settings.deployment
    print(f"\nGenerating {queries_per_doc} query(s) per document using {model}...")

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

    # GPT-5 family rejects non-default temperature; only set it for older models
    create_kwargs: dict = {}
    if not model.startswith("gpt-5"):
        create_kwargs["temperature"] = 0.7

    system_prompt = load_prompt("eval_data_single_doc").format(queries_per_doc=queries_per_doc)

    eval_items = []
    for i, doc in enumerate(docs):
        # Truncate very long documents to fit context
        excerpt = doc["content"][:3000]
        title_hint = f" (from: {doc['title']})" if doc["title"] else ""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document{title_hint}:\n\n{excerpt}"},
                ],
                **create_kwargs,
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
# Multi-document query generation — questions requiring 2+ sources
# ---------------------------------------------------------------------------

def generate_multi_doc_queries(
    docs: list[dict],
    num_pairs: int,
    deployment: str | None = None,
) -> list[dict]:
    """Generate questions that require information from two documents.

    Samples document pairs, feeds both excerpts to GPT, and asks for questions
    that can only be answered by combining information from both sources.
    This tests the retriever's ability to surface multiple relevant documents.
    """
    from openai import AzureOpenAI
    from config import settings

    if len(docs) < 2:
        print("\nSkipping multi-doc generation (need at least 2 documents).")
        return []

    model = deployment or settings.deployment
    print(f"\nGenerating multi-doc queries ({num_pairs} pairs) using {model}...")

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

    create_kwargs: dict = {}
    if not model.startswith("gpt-5"):
        create_kwargs["temperature"] = 0.7

    system_prompt = load_prompt("eval_data_multi_doc")

    # Build pairs — random sampling, avoid duplicates
    all_pairs = []
    doc_indices = list(range(len(docs)))
    for _ in range(num_pairs * 3):  # oversample to account for failures
        if len(all_pairs) >= num_pairs:
            break
        pair = tuple(sorted(random.sample(doc_indices, 2)))
        if pair not in all_pairs:
            all_pairs.append(pair)

    eval_items = []
    for pi, (idx_a, idx_b) in enumerate(all_pairs[:num_pairs]):
        doc_a, doc_b = docs[idx_a], docs[idx_b]
        excerpt_a = doc_a["content"][:2000]
        excerpt_b = doc_b["content"][:2000]

        user_msg = (
            f"Document A (title: {doc_a['title']}):\n\n{excerpt_a}\n\n"
            f"---\n\n"
            f"Document B (title: {doc_b['title']}):\n\n{excerpt_b}"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                **create_kwargs,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[: raw.rfind("```")]
            items = json.loads(raw)

            for item in items[:2]:
                eval_items.append({
                    "query": item["query"],
                    "ground_truth": item.get("ground_truth", ""),
                    "source_doc_ids": [doc_a["id"], doc_b["id"]],
                    "source_doc_titles": [doc_a["title"], doc_b["title"]],
                    "multi_doc": True,
                })

            titles = f"{doc_a['title'][:25]} + {doc_b['title'][:25]}"
            print(f"  [{pi+1}/{num_pairs}] {titles:<55} → {len(items)} queries")

        except Exception as exc:
            print(f"  [{pi+1}/{num_pairs}] FAILED: {exc}")
            continue

        time.sleep(0.5)

    print(f"\nGenerated {len(eval_items)} multi-doc eval queries.")
    return eval_items


# ---------------------------------------------------------------------------
# Retrieval validation — ARES-style filtering
# ---------------------------------------------------------------------------

def filter_queries_by_retrievability(
    client,
    eval_items: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Filter out synthetic queries whose source document(s) are not retrievable.

    For each generated query, runs a search against the same index and checks
    whether the original source document(s) appear in the top-K results.

    - Single-doc queries: source doc must be in top-K
    - Multi-doc queries: ALL source docs must be in top-K

    This mirrors the ARES approach: after generating a synthetic query, verify
    that the original source passage is retrievable; otherwise reject it.
    """
    print(f"\nRetrieval validation (top-{top_k})...")

    kept: list[dict] = []
    discarded = 0

    for i, item in enumerate(eval_items):
        query_text = item["query"]

        # Collect expected IDs — support both single and multi-doc
        if item.get("source_doc_ids"):
            expected_ids = [sid.strip() for sid in item["source_doc_ids"]]
        elif item.get("source_doc_id", "").strip():
            expected_ids = [item["source_doc_id"].strip()]
        else:
            kept.append(item)
            continue

        try:
            results = client.search(search_text=query_text, top=top_k)
            returned_ids = [normalize_doc_id(dict(r)) for r in results]

            missing = [eid for eid in expected_ids if eid not in returned_ids]

            if not missing:
                kept.append(item)
            else:
                discarded += 1
                kind = "multi-doc" if item.get("multi_doc") else "single-doc"
                rank_info = ", ".join(returned_ids[:top_k]) if returned_ids else "(no results)"
                print(
                    f"  DISCARDED: [{i+1}/{len(eval_items)}] "
                    f"{kind} query did not retrieve all source docs in top-{top_k}\n"
                    f"    query:    {query_text[:80]}\n"
                    f"    missing:  {', '.join(missing)}\n"
                    f"    got:      {rank_info}"
                )
        except Exception as exc:
            print(f"  WARNING: [{i+1}/{len(eval_items)}] search failed ({exc}), keeping query")
            kept.append(item)

        time.sleep(0.1)

    total = len(eval_items)
    pct = (len(kept) / total * 100) if total else 0
    print(f"\nRetrieval validation: kept {len(kept)} / {total} queries ({pct:.1f}%)")
    return kept


# ---------------------------------------------------------------------------
# BYOD pipeline — get responses and context for each query
# ---------------------------------------------------------------------------

def run_byod_pipeline(eval_items: list[dict]) -> list[dict]:
    """Run each query through the BYOD pipeline to capture response + context.

    Uses the raw OpenAI SDK (not LangChain) because LangChain's AzureChatOpenAI
    strips the On Your Data citation context from the response. The raw SDK
    exposes it via message.model_extra.context.citations.
    """
    from openai import AzureOpenAI
    from app import get_byod_extra_body
    from config import settings

    print(f"\nRunning {len(eval_items)} queries through BYOD pipeline (raw SDK)...")

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
    extra_body = get_byod_extra_body()

    for i, item in enumerate(eval_items):
        try:
            resp = client.chat.completions.create(
                model=settings.deployment,
                messages=[
                    {"role": "system", "content": load_prompt("helpful_assistant")},
                    {"role": "user", "content": item["query"]},
                ],
                extra_body=extra_body,
                temperature=0.7,
            )

            msg = resp.choices[0].message
            item["response"] = msg.content or ""

            # Extract citations from model_extra (On Your Data context)
            context = ""
            model_extra = msg.model_extra or {}
            ctx_block = model_extra.get("context", {})
            citations = ctx_block.get("citations") or ctx_block.get("documents") or []
            if citations:
                context = "\n\n".join(
                    c.get("content", "") for c in citations if c.get("content")
                )
            item["context"] = context

            status = f"{len(item['response'])} resp / {len(context)} ctx"
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
    gen_group.add_argument(
        "--retrieval-top-k", type=int, default=5,
        help="Top-K search results used to validate synthetic query retrievability (default: 5)",
    )
    gen_group.add_argument(
        "--gen-model", type=str, default=None,
        help="Azure OpenAI deployment for query generation (overrides AZURE_OPENAI_DEPLOYMENT). "
             "Smarter models produce higher-quality synthetic queries and ground-truth answers.",
    )
    gen_group.add_argument(
        "--multi-doc-pairs", type=int, default=5,
        help="Number of document pairs for multi-doc query generation — questions that require "
             "information from 2 documents to answer (default: 5, set to 0 to disable)",
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
    print(f"Retrieval top-K : {args.retrieval_top_k}")
    print(f"Gen model       : {args.gen_model or '(default from .env)'}")
    print(f"Multi-doc pairs : {args.multi_doc_pairs}")
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

    # Step 2: Generate single-doc queries from document content
    eval_items = generate_queries(docs, args.queries_per_doc, deployment=args.gen_model)
    if not eval_items:
        print("ERROR: No queries generated. Check GPT connectivity.")
        sys.exit(1)

    # Step 2b: Generate multi-doc queries (questions requiring 2 sources)
    if args.multi_doc_pairs > 0:
        multi_items = generate_multi_doc_queries(
            docs, num_pairs=args.multi_doc_pairs, deployment=args.gen_model,
        )
        eval_items.extend(multi_items)
        print(f"\nTotal queries: {len(eval_items)} ({len(eval_items) - len(multi_items)} single-doc + {len(multi_items)} multi-doc)")

    # Step 3: Retrieval validation — ARES-style filtering
    eval_items = filter_queries_by_retrievability(
        search_client, eval_items, top_k=args.retrieval_top_k,
    )
    if not eval_items:
        print("ERROR: All queries were discarded by retrieval validation.")
        print("       Try increasing --retrieval-top-k or reviewing query quality.")
        sys.exit(1)

    # Step 4: Run through BYOD pipeline (optional)
    if not args.skip_byod:
        eval_items = run_byod_pipeline(eval_items)

    # Step 5: Write output
    write_jsonl(eval_items, output)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(eval_items)} eval queries ready.")
    print(f"Run evals with: uv run python -m evals.run_all --suite byod")


if __name__ == "__main__":
    main()
