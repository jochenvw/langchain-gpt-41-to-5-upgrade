"""Run evals across multiple GPT-5 family models and produce a comparison table.

Usage:
    python scripts/run_model_comparison.py
"""

import json
import subprocess
import sys
from pathlib import Path

MODELS = ["gpt-5.1", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4", "gpt-5.4-2"]
PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe")
ROOT = Path(__file__).resolve().parent.parent


def run_eval(model: str) -> dict | None:
    """Run eval for a single model, return parsed results."""
    safe = model.replace(".", "_")
    out_file = ROOT / f"eval_results_{safe}.json"

    print(f"\n{'='*60}")
    print(f"  Running eval for: {model}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [PYTHON, "-m", "evals.eval_byod", "--use-foundry", "--model", model],
        cwd=str(ROOT),
        timeout=1800,  # 30 min max per model
    )

    if result.returncode != 0:
        print(f"  *** {model} eval FAILED (exit code {result.returncode}) ***")
        return None

    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def _safe_avg(values):
    """Average of non-None values, or None if empty."""
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean)) if clean else None


def print_comparison(all_results: dict):
    """Print a formatted comparison table."""
    # Load baselines
    baseline_files = {
        "GPT-4.1 BYOD": ROOT / "eval_results_byod.json",
        "GPT-5 Foundry": ROOT / "eval_results_byod_foundry.json",
    }
    for label, path in baseline_files.items():
        if path.exists():
            with open(path, encoding="utf-8") as f:
                all_results[label] = json.load(f)

    metrics_keys = [
        ("Groundedness", "groundedness.groundedness"),
        ("Relevance", "relevance.relevance"),
        ("Coherence", "coherence.coherence"),
        ("Fluency", "fluency.fluency"),
        ("Retrieval", "retrieval.retrieval"),
    ]

    models = list(all_results.keys())
    col_width = 16

    print(f"\n\n{'='*80}")
    print("  MULTI-MODEL COMPARISON — All metrics (20 queries)")
    print(f"{'='*80}\n")

    # Header
    header = f"  {'Metric':<14}"
    for m in models:
        header += f" {m:>{col_width}}"
    print(header)
    print(f"  {'-' * (14 + (col_width + 1) * len(models))}")

    # Quality metrics
    for label, key in metrics_keys:
        row = f"  {label:<14}"
        for m in models:
            data = all_results[m]
            metrics = data.get("metrics", data)
            val = metrics.get(key, "n/a")
            if isinstance(val, (int, float)):
                row += f" {val:>{col_width}.2f}"
            else:
                row += f" {str(val):>{col_width}}"
        print(row)

    print(f"  {'-' * (14 + (col_width + 1) * len(models))}")

    # Latency and token stats from rows
    def _e2e_avg(rows):
        vals = [r.get("outputs.latency_ms", 0) for r in rows]
        return f"{sum(vals) / max(len(vals), 1):,.0f}ms"

    def _e2e_min(rows):
        vals = [r.get("outputs.latency_ms", 0) for r in rows]
        return f"{min(vals, default=0):,.0f}ms"

    def _e2e_max(rows):
        vals = [r.get("outputs.latency_ms", 0) for r in rows]
        return f"{max(vals, default=0):,.0f}ms"

    def _ret_avg(rows):
        vals = [r.get("outputs.retrieve_latency_ms") for r in rows]
        vals = [v for v in vals if v is not None]
        return f"{sum(vals) / len(vals):,.0f}ms" if vals else "n/a"

    def _gen_avg(rows):
        vals = [r.get("outputs.generate_latency_ms") for r in rows]
        vals = [v for v in vals if v is not None]
        return f"{sum(vals) / len(vals):,.0f}ms" if vals else "n/a"

    def _total_tokens(rows):
        vals = [r.get("outputs.generate_total_tokens") or r.get("outputs.total_tokens", 0) for r in rows]
        return f"{sum(vals):,}"

    def _avg_tokens(rows):
        vals = [r.get("outputs.generate_total_tokens") or r.get("outputs.total_tokens", 0) for r in rows]
        return f"{sum(vals) / max(len(vals), 1):,.0f}"

    for stat_label, compute_fn in [
        ("E2E Latency", _e2e_avg),
        ("Min E2E", _e2e_min),
        ("Max E2E", _e2e_max),
        ("Avg Retrieve", _ret_avg),
        ("Avg Generate", _gen_avg),
        ("Total Tokens", _total_tokens),
        ("Avg Tokens", _avg_tokens),
    ]:
        row = f"  {stat_label:<14}"
        for m in models:
            data = all_results[m]
            rows = data.get("rows", [])
            if rows:
                val = compute_fn(rows)
            else:
                val = "n/a"
            row += f" {val:>{col_width}}"
        print(row)

    print()

    # Save comparison as structured JSON (schema v2)
    summary = {"schema_version": 2}
    for m in models:
        data = all_results[m]
        metrics = data.get("metrics", data)
        rows = data.get("rows", [])
        latencies = [r.get("outputs.latency_ms", 0) for r in rows]
        gen_tokens = [r.get("outputs.generate_total_tokens") or r.get("outputs.total_tokens", 0) for r in rows]
        ret_lats = [r.get("outputs.retrieve_latency_ms") for r in rows]
        gen_lats = [r.get("outputs.generate_latency_ms") for r in rows]

        # Detect architecture from presence of per-stage timings
        has_stages = any(v is not None for v in ret_lats)

        # Get model names from first row with data
        gen_model = None
        ret_backend = None
        for r in rows:
            gen_model = gen_model or r.get("outputs.generate_model")
            ret_backend = ret_backend or r.get("outputs.retrieve_backend")

        entry = {
            "architecture": "two-call" if has_stages else "single-call",
            "quality": {
                "groundedness": metrics.get("groundedness.groundedness"),
                "relevance": metrics.get("relevance.relevance"),
                "coherence": metrics.get("coherence.coherence"),
                "fluency": metrics.get("fluency.fluency"),
                "retrieval": metrics.get("retrieval.retrieval"),
            },
            "latency": {
                "e2e_avg_ms": round(sum(latencies) / max(len(latencies), 1)),
            },
            "tokens": {
                "total": sum(gen_tokens),
                "avg": round(sum(gen_tokens) / max(len(gen_tokens), 1)),
                "scope": "generation-only" if has_stages else "full-pipeline",
            },
        }

        if has_stages:
            entry["retrieve"] = {
                "backend": ret_backend or "foundry-kb-retrieve",
                "avg_latency_ms": _safe_avg(ret_lats),
            }
            entry["generate"] = {
                "model": gen_model or "unknown",
                "avg_latency_ms": _safe_avg(gen_lats),
            }
        else:
            entry["model"] = gen_model or m

        summary[m] = entry

    comparison_path = ROOT / "eval_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison saved to: {comparison_path}")


def main():
    all_results = {}
    for model in MODELS:
        data = run_eval(model)
        if data:
            all_results[model] = data
        else:
            print(f"  Skipping {model} in comparison (no results)")

    if all_results:
        print_comparison(all_results)
    else:
        print("No results to compare!")
        sys.exit(1)


if __name__ == "__main__":
    main()
