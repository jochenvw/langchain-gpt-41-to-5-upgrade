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
    for stat_label, compute_fn in [
        ("Avg Latency", lambda rows: f"{sum(r.get('outputs.latency_ms', 0) for r in rows) / max(len(rows), 1):,.0f}ms"),
        ("Min Latency", lambda rows: f"{min((r.get('outputs.latency_ms', 0) for r in rows), default=0):,.0f}ms"),
        ("Max Latency", lambda rows: f"{max((r.get('outputs.latency_ms', 0) for r in rows), default=0):,.0f}ms"),
        ("Total Tokens", lambda rows: f"{sum(r.get('outputs.total_tokens', 0) for r in rows):,}"),
        ("Avg Tokens", lambda rows: f"{sum(r.get('outputs.total_tokens', 0) for r in rows) / max(len(rows), 1):,.0f}"),
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

    # Save comparison as JSON
    summary = {}
    for m in models:
        data = all_results[m]
        metrics = data.get("metrics", data)
        rows = data.get("rows", [])
        latencies = [r.get("outputs.latency_ms", 0) for r in rows]
        tokens = [r.get("outputs.total_tokens", 0) for r in rows]
        summary[m] = {
            "groundedness": metrics.get("groundedness.groundedness"),
            "relevance": metrics.get("relevance.relevance"),
            "coherence": metrics.get("coherence.coherence"),
            "fluency": metrics.get("fluency.fluency"),
            "retrieval": metrics.get("retrieval.retrieval"),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1)),
            "total_tokens": sum(tokens),
            "avg_tokens": round(sum(tokens) / max(len(tokens), 1)),
        }

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
