"""Chat mode evaluations — small baseline.

Runs a handful of general Q&A queries through the LangChain chat pipeline
and evaluates coherence, fluency, and relevance using AI-assisted evaluators
plus F1 NLP score against ground truth.

Usage:
    python -m evals.eval_chat            # from project root
    python evals/eval_chat.py            # direct
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azure.ai.evaluation import (
    CoherenceEvaluator,
    F1ScoreEvaluator,
    FluencyEvaluator,
    RelevanceEvaluator,
    evaluate,
)

from evals.eval_config import DATA_DIR, get_foundry_project, get_model_config


def build_chat_target():
    """Return a callable that invokes the LangChain chat pipeline.

    The callable accepts a dict with 'query' and returns a dict with
    'response' — the contract expected by the evaluate() API.
    """
    from app import build_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm()
    system = SystemMessage(content="You are a helpful assistant.")

    def target_fn(query: str, **kwargs) -> dict:
        messages = [system, HumanMessage(content=query)]

        start = time.perf_counter()
        result = llm.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000

        # Extract token usage from LangChain's usage_metadata
        usage = getattr(result, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        return {
            "response": result.content,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    return target_fn


def main():
    model_config = get_model_config()
    foundry_project = get_foundry_project()
    data_path = str(DATA_DIR / "chat_test_data.jsonl")

    print("=" * 60)
    print("Chat Mode Evaluation — Azure AI Evaluation SDK")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Endpoint: {model_config['azure_endpoint']}")
    print(f"Deploy  : {model_config['azure_deployment']}")
    print(f"Foundry : {'enabled — results will appear in portal' if foundry_project else 'disabled (local only)'}")
    print()

    evaluate_kwargs = dict(
        data=data_path,
        target=build_chat_target(),
        evaluators={
            "coherence": CoherenceEvaluator(model_config),
            "fluency": FluencyEvaluator(model_config),
            "relevance": RelevanceEvaluator(model_config),
            "f1": F1ScoreEvaluator(),
        },
        evaluator_config={
            "default": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${target.response}",
                    "ground_truth": "${data.ground_truth}",
                }
            }
        },
        output_path="./eval_results_chat.json",
    )
    if foundry_project:
        evaluate_kwargs["azure_ai_project"] = foundry_project

    results = evaluate(**evaluate_kwargs)

    print("\n--- Aggregate Scores ---")
    metrics = results.get("metrics", results)
    print(json.dumps(metrics, indent=2))

    # Print latency and token usage summary
    rows = results.get("rows", [])
    if rows:
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

    print(f"\nDetailed results saved to: eval_results_chat.json")


if __name__ == "__main__":
    main()
