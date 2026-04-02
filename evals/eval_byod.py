"""BYOD / RAG evaluation — larger baseline for On Your Data performance.

Runs safety-domain queries through the BYOD pipeline and evaluates:
  - Groundedness  (is the answer grounded in retrieved context?)
  - Relevance     (is the response relevant to the query?)
  - Coherence     (is the response logically consistent?)
  - Fluency       (is the response well-written?)
  - Retrieval     (did the retriever return useful documents?)

This establishes a quality baseline so you can measure impact when
migrating models (e.g. GPT-4.1 → GPT-5) or moving to Foundry Agent Service.

Usage:
    python -m evals.eval_byod            # from project root
    python evals/eval_byod.py            # direct
"""

import json
import sys
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
        result = llm.invoke(messages, extra_body=extra_body)

        # Extract context/citations from response metadata
        context = ""
        if hasattr(result, "response_metadata") and result.response_metadata:
            meta = result.response_metadata
            ctx_block = meta.get("context", {})
            citations = ctx_block.get("citations") or ctx_block.get("documents") or []
            context = "\n\n".join(
                c.get("content", "") for c in citations if c.get("content")
            )

        return {
            "response": result.content,
            "context": context,
        }

    return target_fn


def main():
    model_config = get_model_config()
    foundry_project = get_foundry_project()
    data_path = str(DATA_DIR / "byod_test_data.jsonl")

    print("=" * 60)
    print("BYOD / RAG Evaluation — Azure AI Evaluation SDK")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Endpoint: {model_config['azure_endpoint']}")
    print(f"Deploy  : {model_config['azure_deployment']}")
    print(f"Foundry : {'enabled — results will appear in portal' if foundry_project else 'disabled (local only)'}")
    print()

    evaluate_kwargs = dict(
        data=data_path,
        target=build_byod_target(),
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
        output_path="./eval_results_byod.json",
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
        print(f"\n--- Per-Query Scores ({len(rows)} queries) ---")
        header = f"  {'#':<4} {'Query':<55} {'Ground':>6} {'Rel':>5} {'Coher':>5} {'Flu':>5} {'Retr':>5}"
        print(header)
        print(f"  {'-' * len(header.strip())}")
        for i, row in enumerate(rows):
            q = row.get("inputs.query", f"Q{i+1}")[:55]
            g = row.get("outputs.groundedness.groundedness", "n/a")
            r = row.get("outputs.relevance.relevance", "n/a")
            c = row.get("outputs.coherence.coherence", "n/a")
            f = row.get("outputs.fluency.fluency", "n/a")
            t = row.get("outputs.retrieval.retrieval", "n/a")
            print(f"  [{i+1:<2}] {q:<55} {g:>6} {r:>5} {c:>5} {f:>5} {t:>5}")

    print(f"\nDetailed results saved to: eval_results_byod.json")


if __name__ == "__main__":
    main()
