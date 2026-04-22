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
from pathlib import Path

# Make sibling files (eval_config) and cross-stage code importable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "1-baseline-app"))

from azure.ai.evaluation import (
    CoherenceEvaluator,
    F1ScoreEvaluator,
    FluencyEvaluator,
    RelevanceEvaluator,
    evaluate,
)

from eval_config import DATA_DIR, get_foundry_project, get_judge_model_config, get_model_config


def build_chat_target():
    """Return a callable that invokes the LangChain chat pipeline.

    The callable accepts a dict with 'query' and returns a dict with
    'response' — the contract expected by the evaluate() API.
    """
    from app import build_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm()
    system = SystemMessage(content="You are a helpful assistant.")

    def _target(query: str) -> str:
        messages = [system, HumanMessage(content=query)]
        result = llm.invoke(messages)
        return result.content

    # Wrap to match evaluate() target signature: dict in → dict out
    def target_fn(query: str, **kwargs) -> dict:
        return {"response": _target(query)}

    return target_fn


def main(target_model: str | None = None, judge_model: str | None = None):
    model_config = get_model_config(deployment_override=target_model)
    judge_config = get_judge_model_config(deployment_override=judge_model)
    foundry_project = get_foundry_project()
    data_path = str(DATA_DIR / "chat_test_data.jsonl")

    print("=" * 60)
    print("Chat Mode Evaluation — Azure AI Evaluation SDK")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Target  : {model_config['azure_deployment']} @ {model_config['azure_endpoint']}")
    print(f"Judge   : {judge_config['azure_deployment']} @ {judge_config['azure_endpoint']}")
    print(f"Foundry : {'enabled — results will appear in portal' if foundry_project else 'disabled (local only)'}")
    print()

    evaluate_kwargs = dict(
        data=data_path,
        target=build_chat_target(),
        evaluators={
            "coherence": CoherenceEvaluator(judge_config),
            "fluency": FluencyEvaluator(judge_config),
            "relevance": RelevanceEvaluator(judge_config),
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
    print(f"\nDetailed results saved to: eval_results_chat.json")


if __name__ == "__main__":
    main()
