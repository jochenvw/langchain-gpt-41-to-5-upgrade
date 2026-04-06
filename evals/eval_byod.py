"""BYOD / RAG evaluation — larger baseline for On Your Data performance.

Runs safety-domain queries through the BYOD pipeline and evaluates:
  - Groundedness  (is the answer grounded in retrieved context?)
  - Relevance     (is the response relevant to the query?)
  - Coherence     (is the response logically consistent?)
  - Fluency       (is the response well-written?)
  - Retrieval     (did the retriever return useful documents?)

Supports two backends:
  - Legacy BYOD (On Your Data) — default, uses GPT-4.1
  - GPT-5 RAG (Responses API + direct search) — use --use-foundry

Usage:
    python -m evals.eval_byod                    # legacy BYOD
    python -m evals.eval_byod --use-foundry      # GPT-5 RAG
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azure.ai.evaluation import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    RetrievalEvaluator,
    evaluate,
)

from evals.eval_config import DATA_DIR, get_foundry_project, get_model_config


def build_byod_target():
    """Return a callable that invokes the BYOD (On Your Data) pipeline.

    Captures both the response text and the context/citations returned
    by Azure AI Search so groundedness and retrieval can be evaluated.
    """
    from app import build_llm, get_byod_extra_body
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm()
    extra_body = get_byod_extra_body()
    system = SystemMessage(content="You are a helpful assistant.")

    def target_fn(query: str, **kwargs) -> dict:
        messages = [system, HumanMessage(content=query)]

        start = time.perf_counter()
        result = llm.invoke(messages, extra_body=extra_body)
        latency_ms = (time.perf_counter() - start) * 1000

        # Extract context/citations preserved by AzureChatOpenAIWithContext
        context = ""
        ctx_block = (result.additional_kwargs or {}).get("context", {})
        citations = ctx_block.get("citations") or ctx_block.get("documents") or []
        if citations:
            context = "\n\n".join(
                c.get("content", "") for c in citations if c.get("content")
            )

        # Extract token usage from LangChain's usage_metadata
        usage = getattr(result, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        return {
            "response": result.content,
            "context": context,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    return target_fn


def build_foundry_target():
    """Return a callable that invokes GPT-5 RAG via Responses API + direct search.

    Queries Azure AI Search directly, injects context, and calls GPT-5 via
    the OpenAI Responses API. Replaces the deprecated BYOD/On Your Data pipeline.
    """
    from app import query_foundry_rag

    print("  GPT-5 RAG target: Responses API + direct Azure AI Search")

    def target_fn(query: str, **kwargs) -> dict:
        start = time.perf_counter()
        result = query_foundry_rag(query)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "response": result.get("response", ""),
            "context": result.get("context", ""),
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return target_fn


def main(use_foundry: bool = False):
    model_config = get_model_config()
    foundry_project = get_foundry_project()
    data_path = str(DATA_DIR / "byod_test_data.jsonl")

    backend = "GPT-5 RAG (Responses API + Azure AI Search)" if use_foundry else "Legacy BYOD (On Your Data)"

    print("=" * 60)
    print(f"RAG Evaluation — {backend}")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    if not use_foundry:
        print(f"Endpoint: {model_config['azure_endpoint']}")
        print(f"Deploy  : {model_config['azure_deployment']}")
    print(f"Foundry : {'enabled — results will appear in portal' if foundry_project else 'disabled (local only)'}")
    print()

    target = build_foundry_target() if use_foundry else build_byod_target()

    output_path = "./eval_results_byod_foundry.json" if use_foundry else "./eval_results_byod.json"

    evaluate_kwargs = dict(
        data=data_path,
        target=target,
        evaluators={
            "groundedness": GroundednessEvaluator(model_config),
            "relevance": RelevanceEvaluator(model_config),
            "coherence": CoherenceEvaluator(model_config),
            "fluency": FluencyEvaluator(model_config),
            "retrieval": RetrievalEvaluator(model_config),
        },
        evaluator_config={
            "default": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${target.response}",
                    "context": "${target.context}",
                }
            }
        },
        output_path=output_path,
    )
    if foundry_project:
        evaluate_kwargs["azure_ai_project"] = foundry_project

    results = evaluate(**evaluate_kwargs)

    print("\n--- Aggregate Scores ---")
    metrics = results.get("metrics", results)
    print(json.dumps(metrics, indent=2))

    # Print per-query breakdown if available
    rows = results.get("rows", [])
    if rows:
        # Latency & token summary
        latencies = [r.get("outputs.latency_ms", 0) for r in rows if r.get("outputs.latency_ms")]
        total_toks = [r.get("outputs.total_tokens", 0) for r in rows if r.get("outputs.total_tokens")]
        if latencies:
            print(f"\n--- Latency & Token Usage ({len(rows)} queries) ---")
            print(f"  Avg latency   : {sum(latencies)/len(latencies):.0f} ms")
            print(f"  Min latency   : {min(latencies):.0f} ms")
            print(f"  Max latency   : {max(latencies):.0f} ms")
        if total_toks:
            print(f"  Avg tokens    : {sum(total_toks)/len(total_toks):.0f}")
            print(f"  Total tokens  : {sum(total_toks)}")

        print(f"\n--- Per-Query Scores ({len(rows)} queries) ---")
        header = f"  {'#':<4} {'Query':<45} {'Ground':>6} {'Rel':>5} {'Coher':>5} {'Flu':>5} {'Retr':>5} {'ms':>7} {'Tokens':>6}"
        print(header)
        print(f"  {'-' * len(header.strip())}")
        for i, row in enumerate(rows):
            q = row.get("inputs.query", f"Q{i+1}")[:45]
            g = row.get("outputs.groundedness.groundedness", "n/a")
            r = row.get("outputs.relevance.relevance", "n/a")
            c = row.get("outputs.coherence.coherence", "n/a")
            f = row.get("outputs.fluency.fluency", "n/a")
            t = row.get("outputs.retrieval.retrieval", "n/a")
            ms = row.get("outputs.latency_ms", "n/a")
            tok = row.get("outputs.total_tokens", "n/a")
            print(f"  [{i+1:<2}] {q:<45} {g:>6} {r:>5} {c:>5} {f:>5} {t:>5} {ms:>7} {tok:>6}")

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-foundry", action="store_true", help="Use GPT-5 RAG via Responses API instead of BYOD")
    args = parser.parse_args()
    main(use_foundry=args.use_foundry)
