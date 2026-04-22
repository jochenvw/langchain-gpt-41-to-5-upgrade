"""BYOD / RAG evaluation — larger baseline for On Your Data performance.

Runs safety-domain queries through the BYOD pipeline and evaluates:
  - Groundedness  (is the answer grounded in retrieved context?)
  - Relevance     (is the response relevant to the query?)
  - Coherence     (is the response logically consistent?)
  - Fluency       (is the response well-written?)
  - Retrieval     (did the retriever return useful documents?)

Also captures performance metrics:
  - Response time per query
  - Time-to-first-token (TTFT)
  - Response length (word count)

This establishes a quality baseline so you can measure impact when
migrating models (e.g. GPT-4.1 → GPT-5) or moving to Foundry Agent Service.

Usage:
    python 3-baseline-eval/eval_byod.py            # direct
    python 3-baseline-eval/run.py --suite byod     # via runner
"""

import json
import sys
import time
from pathlib import Path

# Make sibling files (eval_config) and cross-stage code importable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "1-baseline-app"))

from azure.ai.evaluation import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    RetrievalEvaluator,
    evaluate,
)

from eval_config import DATA_DIR, get_foundry_project, get_judge_model_config, get_model_config


def build_byod_target():
    """Return a callable that invokes the BYOD (On Your Data) pipeline.

    Captures both the response text and the context/citations returned
    by Azure AI Search so groundedness and retrieval can be evaluated.
    Also captures performance metrics (response time, TTFT, word count).
    """
    from app import build_llm, get_byod_extra_body
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm()
    extra_body = get_byod_extra_body()
    system = SystemMessage(content="You are a helpful assistant.")

    def target_fn(query: str, **kwargs) -> dict:
        messages = [system, HumanMessage(content=query)]

        t_start = time.perf_counter()
        result = llm.invoke(messages, extra_body=extra_body)
        t_end = time.perf_counter()

        response_time = t_end - t_start
        response_text = result.content or ""
        word_count = len(response_text.split())

        # Extract context/citations from response metadata
        context = ""
        if hasattr(result, "response_metadata") and result.response_metadata:
            meta = result.response_metadata
            ctx_block = meta.get("context", {})
            citations = ctx_block.get("citations") or ctx_block.get("documents") or []
            context = "\n\n".join(
                c.get("content", "") for c in citations if c.get("content")
            )

        # TTFT: use token_usage if available, otherwise approximate
        ttft = None
        if hasattr(result, "response_metadata") and result.response_metadata:
            usage = result.response_metadata.get("token_usage", {})
            # LangChain doesn't expose TTFT directly; approximate as fraction
            # of response time proportional to first token
            if usage.get("completion_tokens", 0) > 0:
                ttft = response_time / max(usage["completion_tokens"], 1)

        return {
            "response": response_text,
            "context": context,
            "response_time_s": round(response_time, 3),
            "ttft_s": round(ttft, 4) if ttft else None,
            "word_count": word_count,
        }

    return target_fn


def main(target_model: str | None = None, judge_model: str | None = None):
    model_config = get_model_config(deployment_override=target_model)
    judge_config = get_judge_model_config(deployment_override=judge_model)
    foundry_project = get_foundry_project()
    data_path = str(DATA_DIR / "byod_test_data.jsonl")

    print("=" * 60)
    print("BYOD / RAG Evaluation — Azure AI Evaluation SDK")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Target  : {model_config['azure_deployment']} @ {model_config['azure_endpoint']}")
    print(f"Judge   : {judge_config['azure_deployment']} @ {judge_config['azure_endpoint']}")
    print(f"Foundry : {'enabled — results will appear in portal' if foundry_project else 'disabled (local only)'}")
    print()

    eval_start = time.perf_counter()

    evaluate_kwargs = dict(
        data=data_path,
        target=build_byod_target(),
        evaluators={
            "groundedness": GroundednessEvaluator(judge_config),
            "relevance": RelevanceEvaluator(judge_config),
            "coherence": CoherenceEvaluator(judge_config),
            "fluency": FluencyEvaluator(judge_config),
            "retrieval": RetrievalEvaluator(judge_config),
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

    eval_end = time.perf_counter()
    total_eval_time = eval_end - eval_start

    print("\n--- Aggregate Scores ---")
    metrics = results.get("metrics", results)
    print(json.dumps(metrics, indent=2))

    # Compute performance metrics from per-query results
    rows = results.get("rows", [])
    if rows:
        response_times = [r.get("outputs.response_time_s", 0) for r in rows if r.get("outputs.response_time_s")]
        ttfts = [r.get("outputs.ttft_s", 0) for r in rows if r.get("outputs.ttft_s")]
        word_counts = [r.get("outputs.word_count", 0) for r in rows if r.get("outputs.word_count")]

        print(f"\n--- Performance Metrics ---")
        if response_times:
            avg_rt = sum(response_times) / len(response_times)
            print(f"  Avg response time     : {avg_rt:.2f}s")
        if ttfts:
            avg_ttft = sum(ttfts) / len(ttfts)
            print(f"  Avg time-to-first-tkn : {avg_ttft:.4f}s")
        if word_counts:
            avg_wc = sum(word_counts) / len(word_counts)
            print(f"  Avg response length   : {avg_wc:.0f} words")
        print(f"  Overall eval time     : {total_eval_time:.1f}s")

        print(f"\n--- Per-Query Scores ({len(rows)} queries) ---")
        header = f"  {'#':<4} {'Query':<45} {'Ground':>6} {'Rel':>5} {'Coher':>5} {'Flu':>5} {'Retr':>5} {'Time':>6} {'Words':>5}"
        print(header)
        print(f"  {'-' * len(header.strip())}")
        for i, row in enumerate(rows):
            q = row.get("inputs.query", f"Q{i+1}")[:45]
            g = row.get("outputs.groundedness.groundedness", "n/a")
            r = row.get("outputs.relevance.relevance", "n/a")
            c = row.get("outputs.coherence.coherence", "n/a")
            f = row.get("outputs.fluency.fluency", "n/a")
            t = row.get("outputs.retrieval.retrieval", "n/a")
            rt = row.get("outputs.response_time_s", "")
            wc = row.get("outputs.word_count", "")
            rt_str = f"{rt:.1f}s" if isinstance(rt, (int, float)) else "n/a"
            wc_str = str(wc) if wc else "n/a"
            print(f"  [{i+1:<2}] {q:<45} {g:>6} {r:>5} {c:>5} {f:>5} {t:>5} {rt_str:>6} {wc_str:>5}")

    print(f"\nDetailed results saved to: eval_results_byod.json")


if __name__ == "__main__":
    main()
