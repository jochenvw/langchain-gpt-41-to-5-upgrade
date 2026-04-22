"""Multi-model comparison experiment.

Runs all GPT-5 family models + the GPT-4.1 BYOD baseline through the eval
pipeline, caches retrieval (it's model-agnostic), and produces a comparison
table with split retrieve/generate timing plus quality scores.

Usage:
    uv run python 5-experiments/run_comparison.py
    uv run python 5-experiments/run_comparison.py --models gpt-5.4-mini gpt-5.4-nano
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "4-two-step-app"))
sys.path.insert(0, str(_ROOT / "3-baseline-eval"))
sys.path.insert(0, str(_ROOT / "1-baseline-app"))

from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from retrieve import retrieve, RetrieveResult
from generate import generate

# ---------------------------------------------------------------------------
# Models to sweep
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-2",
    "gpt-5.1",
    "gpt-5.4",
    "gpt-5-mini",
]

BASELINE_LABEL = "GPT-4.1 BYOD"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_avg(values: list) -> float | None:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(clean) / len(clean), 2) if clean else None


def _fmt(val, suffix="", decimals=0):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:,.{decimals}f}{suffix}"
    return f"{val:,}{suffix}"


def load_queries(data_path: Path) -> list[dict]:
    queries = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Phase 1: Cached retrieval (run once for all models)
# ---------------------------------------------------------------------------

def run_retrieval(queries: list[dict], credential: AzureCliCredential) -> list[dict]:
    """Retrieve documents for each query. Returns list of {query, retrieve_result}."""
    search_ep = os.environ.get("AZURE_SEARCH_ENDPOINT", "").strip()
    if not search_ep:
        print("ERROR: AZURE_SEARCH_ENDPOINT not set"); sys.exit(1)

    print("\n" + "=" * 70)
    print("PHASE 1: Retrieval (cached, model-agnostic)")
    print("=" * 70)

    cached = []
    for i, q in enumerate(queries):
        query_text = q["query"]
        print(f"  [{i+1}/{len(queries)}] Retrieving: {query_text[:60]}...", end=" ", flush=True)
        try:
            result = retrieve(query_text, search_ep, credential)
            print(f"✓ {result.latency_ms:.0f}ms, {len(result.documents)} docs")
            cached.append({"query": query_text, "retrieve_result": result})
        except Exception as exc:
            print(f"✗ {exc}")
            cached.append({"query": query_text, "retrieve_result": None})

    latencies = [c["retrieve_result"].latency_ms for c in cached if c["retrieve_result"]]
    if latencies:
        print(f"\n  Avg retrieve: {sum(latencies)/len(latencies):.0f}ms "
              f"(min={min(latencies):.0f}, max={max(latencies):.0f})")

    return cached


# ---------------------------------------------------------------------------
# Phase 2: Generation per model
# ---------------------------------------------------------------------------

def run_generation_for_model(
    model: str,
    cached_retrieval: list[dict],
    credential: AzureCliCredential,
) -> list[dict]:
    """Run generation for one model across all queries."""
    openai_ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if openai_ep.endswith("/openai"):
        openai_ep = openai_ep[: -len("/openai")]

    results = []
    for i, item in enumerate(cached_retrieval):
        query_text = item["query"]
        ret = item["retrieve_result"]

        if ret is None:
            results.append({
                "query": query_text, "response": "", "context": "",
                "retrieve_latency_ms": None, "generate_latency_ms": None,
                "e2e_latency_ms": None, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
            })
            continue

        print(f"  [{i+1}/{len(cached_retrieval)}] {query_text[:55]}...", end=" ", flush=True)
        try:
            gen = generate(query_text, ret.context_text, model, openai_ep, credential)
            e2e = ret.latency_ms + gen.latency_ms
            resp_preview = (gen.response or "")[:80].replace("\n", " ")
            print(f"✓ {gen.latency_ms:.0f}ms [{gen.total_tokens} tok] {resp_preview}")
            results.append({
                "query": query_text,
                "response": gen.response,
                "context": ret.context_text,
                "retrieve_latency_ms": ret.latency_ms,
                "generate_latency_ms": gen.latency_ms,
                "e2e_latency_ms": round(e2e, 1),
                "generate_model": gen.model,
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "total_tokens": gen.total_tokens,
            })
        except Exception as exc:
            print(f"✗ {exc}")
            results.append({
                "query": query_text, "response": "", "context": ret.context_text,
                "retrieve_latency_ms": ret.latency_ms, "generate_latency_ms": None,
                "e2e_latency_ms": None, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
            })

    return results


# ---------------------------------------------------------------------------
# Phase 3: BYOD baseline
# ---------------------------------------------------------------------------

def run_byod_baseline(queries: list[dict]) -> list[dict]:
    """Run the GPT-4.1 BYOD baseline for comparison."""
    print(f"\n  Running {BASELINE_LABEL}...")

    from app import build_llm, get_byod_extra_body
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm()
    extra_body = get_byod_extra_body()
    system = SystemMessage(content="You are a helpful assistant.")

    results = []
    for i, q in enumerate(queries):
        query_text = q["query"]
        print(f"  [{i+1}/{len(queries)}] {query_text[:55]}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            result = llm.invoke(
                [system, HumanMessage(content=query_text)],
                extra_body=extra_body,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            response_text = result.content or ""
            context = ""
            if hasattr(result, "response_metadata") and result.response_metadata:
                ctx_block = result.response_metadata.get("context", {})
                citations = ctx_block.get("citations") or ctx_block.get("documents") or []
                context = "\n\n".join(c.get("content", "") for c in citations if c.get("content"))

            usage = (result.response_metadata or {}).get("token_usage", {})
            print(f"✓ {latency_ms:.0f}ms [{usage.get('total_tokens', 0)} tok]")
            results.append({
                "query": query_text,
                "response": response_text,
                "context": context,
                "retrieve_latency_ms": None,
                "generate_latency_ms": None,
                "e2e_latency_ms": round(latency_ms, 1),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
        except Exception as exc:
            print(f"✗ {exc}")
            results.append({
                "query": query_text, "response": "", "context": "",
                "retrieve_latency_ms": None, "generate_latency_ms": None,
                "e2e_latency_ms": None, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
            })

    return results


# ---------------------------------------------------------------------------
# Phase 4: Evaluate quality with azure-ai-evaluation SDK
# ---------------------------------------------------------------------------

def evaluate_quality(label: str, query_results: list[dict], judge_config: dict) -> dict:
    """Score a set of query results using the AI eval SDK."""
    from azure.ai.evaluation import (
        CoherenceEvaluator,
        FluencyEvaluator,
        GroundednessEvaluator,
        RelevanceEvaluator,
        RetrievalEvaluator,
    )

    evaluators = {
        "groundedness": GroundednessEvaluator(judge_config),
        "relevance": RelevanceEvaluator(judge_config),
        "coherence": CoherenceEvaluator(judge_config),
        "fluency": FluencyEvaluator(judge_config),
        "retrieval": RetrievalEvaluator(judge_config),
    }

    scores = {name: [] for name in evaluators}

    for i, row in enumerate(query_results):
        if not row["response"]:
            for name in evaluators:
                scores[name].append(None)
            continue

        for name, ev in evaluators.items():
            try:
                result = ev(
                    query=row["query"],
                    response=row["response"],
                    context=row["context"],
                )
                val = result.get(name)
                scores[name].append(val)
            except Exception:
                scores[name].append(None)

    return {name: _safe_avg(vals) for name, vals in scores.items()}


# ---------------------------------------------------------------------------
# Phase 5: Print comparison table + save JSON
# ---------------------------------------------------------------------------

def print_comparison(all_results: dict[str, dict], output_dir: Path):
    """Print the comparison table and save results."""

    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    # Header
    hdr = (
        f"{'Model':<20} {'E2E':>8} {'Retrieve':>10} {'Generate':>10} "
        f"{'Ground':>7} {'Relev':>7} {'Coher':>7} {'Fluency':>7} {'Retr':>7} "
        f"{'Tokens':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    comparison_json = {"schema_version": 2}

    for label, data in all_results.items():
        rows = data["rows"]
        quality = data["quality"]

        e2e_vals = [r["e2e_latency_ms"] for r in rows if r.get("e2e_latency_ms")]
        ret_vals = [r["retrieve_latency_ms"] for r in rows if r.get("retrieve_latency_ms")]
        gen_vals = [r["generate_latency_ms"] for r in rows if r.get("generate_latency_ms")]
        tok_vals = [r["total_tokens"] for r in rows if r.get("total_tokens")]

        e2e_avg = _safe_avg(e2e_vals)
        ret_avg = _safe_avg(ret_vals)
        gen_avg = _safe_avg(gen_vals)
        tok_avg = _safe_avg(tok_vals)
        tok_total = sum(tok_vals)
        non_empty = sum(1 for r in rows if r.get("response"))

        print(
            f"{label:<20} "
            f"{_fmt(e2e_avg, 'ms'):>8} "
            f"{_fmt(ret_avg, 'ms'):>10} "
            f"{_fmt(gen_avg, 'ms'):>10} "
            f"{_fmt(quality.get('groundedness'), decimals=2):>7} "
            f"{_fmt(quality.get('relevance'), decimals=2):>7} "
            f"{_fmt(quality.get('coherence'), decimals=2):>7} "
            f"{_fmt(quality.get('fluency'), decimals=2):>7} "
            f"{_fmt(quality.get('retrieval'), decimals=2):>7} "
            f"{_fmt(tok_total):>8}"
        )

        is_byod = label == BASELINE_LABEL
        entry = {
            "architecture": "single-call" if is_byod else "two-call",
            "queries_run": len(rows),
            "queries_answered": non_empty,
            "quality": quality,
            "latency": {
                "e2e_avg_ms": e2e_avg,
                "e2e_min_ms": round(min(e2e_vals), 1) if e2e_vals else None,
                "e2e_max_ms": round(max(e2e_vals), 1) if e2e_vals else None,
            },
            "tokens": {"total": tok_total, "avg": tok_avg, "scope": "full-pipeline" if is_byod else "generation-only"},
        }
        if not is_byod:
            entry["retrieve"] = {"backend": "foundry-kb-retrieve", "avg_latency_ms": ret_avg}
            entry["generate"] = {"model": label, "avg_latency_ms": gen_avg}

        comparison_json[label] = entry

        # Save per-model results
        out_file = output_dir / f"eval_results_{label.lower().replace(' ', '_').replace('.', '_')}.json"
        with open(out_file, "w") as f:
            json.dump({"model": label, "rows": rows, "quality": quality}, f, indent=2)
        print(f"  → saved {out_file.name}")

    # Save comparison
    comp_file = output_dir / "eval_comparison.json"
    with open(comp_file, "w") as f:
        json.dump(comparison_json, f, indent=2)
    print(f"\n  → Comparison saved to {comp_file}")

    # Recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    baseline_data = all_results.get(BASELINE_LABEL)
    if not baseline_data:
        print("  No BYOD baseline to compare against.")
        return

    bl_quality = baseline_data["quality"]
    bl_e2e = _safe_avg([r["e2e_latency_ms"] for r in baseline_data["rows"] if r.get("e2e_latency_ms")])

    print(f"\n  Baseline ({BASELINE_LABEL}): E2E={_fmt(bl_e2e, 'ms')}, "
          f"Groundedness={_fmt(bl_quality.get('groundedness'), decimals=2)}")

    candidates = []
    for label, data in all_results.items():
        if label == BASELINE_LABEL:
            continue
        q = data["quality"]
        rows = data["rows"]
        e2e = _safe_avg([r["e2e_latency_ms"] for r in rows if r.get("e2e_latency_ms")])
        answered = sum(1 for r in rows if r.get("response"))
        if e2e is None or answered < len(rows) * 0.7:
            continue
        ground = q.get("groundedness") or 0
        if bl_quality.get("groundedness") and ground >= bl_quality["groundedness"] * 0.9:
            candidates.append((label, e2e, q, data))

    if not candidates:
        print("  No model meets 90% of baseline groundedness. Consider gpt-5.4-mini for best latency.\n")
        return

    candidates.sort(key=lambda x: x[1])
    best_label, best_e2e, best_q, _ = candidates[0]
    delta_pct = ((best_e2e - bl_e2e) / bl_e2e * 100) if bl_e2e else 0

    print(f"\n  ★ Recommended: {best_label}")
    print(f"    E2E: {_fmt(best_e2e, 'ms')} ({delta_pct:+.0f}% vs baseline)")
    print(f"    Groundedness: {_fmt(best_q.get('groundedness'), decimals=2)} | "
          f"Relevance: {_fmt(best_q.get('relevance'), decimals=2)} | "
          f"Coherence: {_fmt(best_q.get('coherence'), decimals=2)}")

    if len(candidates) > 1:
        print(f"\n  Runner-up options (within 90% groundedness):")
        for label, e2e, q, _ in candidates[1:3]:
            d = ((e2e - bl_e2e) / bl_e2e * 100) if bl_e2e else 0
            print(f"    - {label}: E2E={_fmt(e2e, 'ms')} ({d:+.0f}%), "
                  f"Ground={_fmt(q.get('groundedness'), decimals=2)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-model comparison experiment")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, help="Models to test")
    parser.add_argument("--skip-byod", action="store_true", help="Skip the BYOD baseline")
    parser.add_argument("--judge", default="gpt-4.1", help="Judge model for quality eval")
    args = parser.parse_args()

    data_path = _ROOT / "2-eval-data" / "data" / "byod_test_data.jsonl"
    queries = load_queries(data_path)
    print(f"Loaded {len(queries)} queries from {data_path}")

    credential = AzureCliCredential()

    # Judge config
    from eval_config import get_judge_model_config
    judge_config = get_judge_model_config(deployment_override=args.judge)
    print(f"Judge model: {args.judge} @ {judge_config['azure_endpoint']}")

    # Phase 1: Retrieval (once for all models)
    cached = run_retrieval(queries, credential)

    all_results: dict[str, dict] = {}

    # Phase 2: BYOD baseline
    if not args.skip_byod:
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: BYOD Baseline ({BASELINE_LABEL})")
        print("=" * 70)
        byod_rows = run_byod_baseline(queries)
        print(f"\n  Evaluating quality...")
        byod_quality = evaluate_quality(BASELINE_LABEL, byod_rows, judge_config)
        all_results[BASELINE_LABEL] = {"rows": byod_rows, "quality": byod_quality}
        print(f"  Quality: {byod_quality}")

    # Phase 3: GPT-5 models
    for model in args.models:
        print(f"\n{'=' * 70}")
        print(f"PHASE 3: Generation — {model}")
        print("=" * 70)
        gen_rows = run_generation_for_model(model, cached, credential)

        # Save partial results immediately
        partial_file = _ROOT / f"eval_results_{model.replace('.', '_')}_partial.json"
        with open(partial_file, "w") as f:
            json.dump({"model": model, "rows": gen_rows}, f, indent=2)

        print(f"\n  Evaluating quality...")
        quality = evaluate_quality(model, gen_rows, judge_config)
        all_results[model] = {"rows": gen_rows, "quality": quality}
        print(f"  Quality: {quality}")

        # Clean up partial
        partial_file.unlink(missing_ok=True)

    # Phase 4: Comparison
    output_dir = _ROOT
    print_comparison(all_results, output_dir)


if __name__ == "__main__":
    main()
