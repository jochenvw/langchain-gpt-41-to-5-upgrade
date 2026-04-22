"""Run BYOD-style evals against the two-step app (4-two-step-app) for a given model.

This is the stage-5 counterpart to 3-baseline-eval/run.py.
- 3-baseline-eval/  → scores 1-baseline-app (GPT-4.1 + BYOD)
- 5-experiments/    → scores 4-two-step-app (configurable model + retrieval)

Usage:
    python 5-experiments/run.py                           # current FOUNDRY_MODEL_DEPLOYMENT
    python 5-experiments/run.py --model gpt-5-mini        # override model
    python 5-experiments/run.py --reasoning-effort low    # tune generator
    python 5-experiments/run.py --max-context-chars 8000  # tune retriever
    python 5-experiments/run.py --output results/gpt-5-mini.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# sys.path: this dir + repo root + sibling stages we import from
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "4-two-step-app"))
sys.path.insert(0, str(_ROOT / "3-baseline-eval"))

from azure.ai.evaluation import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    RetrievalEvaluator,
    evaluate,
)

from eval_config import DATA_DIR, get_foundry_project, get_judge_model_config


def build_two_step_target(
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    retrieve_reasoning_effort: str = "minimal",
    reranker_threshold: float = 1.5,
    max_context_chars: int = 15_000,
):
    """Return a callable matching the 3-baseline-eval contract.

    Reuses Retriever + Generator across all queries (connection pooling,
    cached AAD token).
    """
    from app import TwoStepSession  # 4-two-step-app/app.py
    from retrieve import Retriever
    from generate import Generator

    session = TwoStepSession(
        retriever=Retriever(
            reasoning_effort=retrieve_reasoning_effort,
            reranker_threshold=reranker_threshold,
            max_context_chars=max_context_chars,
        ),
        generator=Generator(
            model=model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
        ),
    )

    def target_fn(query: str, **_kwargs) -> dict:
        result = session.query(query)
        words = len((result.response or "").split())
        ret_ms = result.retrieve_latency_ms or 0
        gen_ms = result.generate_latency_ms or 0
        total_tokens = (result.generate_prompt_tokens or 0) + (result.generate_completion_tokens or 0)
        # Field names match what sweep.py / historical EVAL_RESULTS aggregators expect.
        return {
            "response": result.response,
            "context": result.context,
            "ground_truth": "",
            "latency_ms": round(ret_ms + gen_ms),
            "retrieve_latency_ms": round(ret_ms) if result.retrieve_latency_ms is not None else None,
            "generate_latency_ms": round(gen_ms) if result.generate_latency_ms is not None else None,
            "ttft_s": None,
            "response_words": words,
            "generate_model": result.generate_model,
            "retrieve_backend": "foundry-kb-retrieve",
            "generate_prompt_tokens": result.generate_prompt_tokens,
            "generate_completion_tokens": result.generate_completion_tokens,
            "generate_total_tokens": total_tokens or None,
        }

    return target_fn


def main(
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    retrieve_reasoning_effort: str = "minimal",
    reranker_threshold: float = 1.5,
    max_context_chars: int = 15_000,
    judge_model: str | None = None,
    output: str | None = None,
) -> None:
    judge_cfg = get_judge_model_config(deployment_override=judge_model)
    data_path = str(DATA_DIR / "byod_test_data.jsonl")

    target = build_two_step_target(
        model=model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        retrieve_reasoning_effort=retrieve_reasoning_effort,
        reranker_threshold=reranker_threshold,
        max_context_chars=max_context_chars,
    )

    evaluators = {
        "groundedness": GroundednessEvaluator(model_config=judge_cfg),
        "relevance": RelevanceEvaluator(model_config=judge_cfg),
        "coherence": CoherenceEvaluator(model_config=judge_cfg),
        "fluency": FluencyEvaluator(model_config=judge_cfg),
        "retrieval": RetrievalEvaluator(model_config=judge_cfg),
    }

    label = f"two-step-{model or 'default'}"
    out_path = output or str(_HERE / "results" / f"twostep_{(model or 'default').replace('.', '_')}.json")

    print(f"\n=== Stage 5 eval — {label} ===")
    print(f"Data       : {data_path}")
    print(f"Output     : {out_path}")
    print(f"Judge      : {judge_cfg.get('azure_deployment')}")
    print(f"Generator  : model={model or '(default)'} effort={reasoning_effort} max_tokens={max_completion_tokens}")
    print(f"Retriever  : effort={retrieve_reasoning_effort} reranker={reranker_threshold} max_ctx={max_context_chars}")

    t0 = time.perf_counter()
    result = evaluate(
        data=data_path,
        target=target,
        evaluators=evaluators,
        evaluator_config={
            ev: {"column_mapping": {
                "query": "${data.query}",
                "response": "${target.response}",
                "context": "${target.context}",
                "ground_truth": "${data.ground_truth}",
            }} for ev in evaluators
        },
        azure_ai_project=get_foundry_project(),
        output_path=out_path,
    )
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\nDone in {elapsed}s. Results -> {out_path}")
    print(f"Studio URL: {result.get('studio_url') if isinstance(result, dict) else 'n/a'}")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Stage-5 two-step app eval runner")
    p.add_argument("--model", default=None,
                   help="Override FOUNDRY_MODEL_DEPLOYMENT for the generator")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["low", "medium", "high"],
                   help="Generator reasoning effort (model-dependent)")
    p.add_argument("--max-completion-tokens", type=int, default=None)
    p.add_argument("--retrieve-reasoning-effort", default="minimal",
                   choices=["minimal", "low", "medium", "high"])
    p.add_argument("--reranker-threshold", type=float, default=1.5)
    p.add_argument("--max-context-chars", type=int, default=15_000)
    p.add_argument("--judge-model", default=None,
                   help="Override AZURE_EVAL_DEPLOYMENT (judge model)")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: results/twostep_<model>.json)")
    args = p.parse_args()
    main(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        retrieve_reasoning_effort=args.retrieve_reasoning_effort,
        reranker_threshold=args.reranker_threshold,
        max_context_chars=args.max_context_chars,
        judge_model=args.judge_model,
        output=args.output,
    )


if __name__ == "__main__":
    _cli()
