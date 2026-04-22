# 5 — Experiments

Sweep the two-step app (stage 4) across model + retrieval configurations.
Goal: find the cheapest setup that meets or beats the baseline (stage 3).

## Files

```
run.py        Single-run eval against 4-two-step-app (parameterized)
sweep.py      Loops run.py over multiple models, builds comparison.json
results/      Per-run JSON + comparison.json + historical baselines
RESULTS.md    Findings from a previous full sweep (reference)
```

## Run a single model

```bash
uv run python 5-experiments/run.py --model gpt-5
uv run python 5-experiments/run.py --model gpt-5-mini --reasoning-effort low
uv run python 5-experiments/run.py --model gpt-5.4 --max-completion-tokens 2000

# Tune the retriever side too
uv run python 5-experiments/run.py --model gpt-5 \
    --retrieve-reasoning-effort low \
    --reranker-threshold 2.0 \
    --max-context-chars 8000
```

## Sweep across models

```bash
# Default sweep: gpt-5, gpt-5-mini, gpt-5.1, gpt-5.4-{nano,mini,full}
uv run python 5-experiments/sweep.py

# Custom model list
uv run python 5-experiments/sweep.py --models gpt-5-mini,gpt-5.4-mini

# All models at low reasoning
uv run python 5-experiments/sweep.py --reasoning-effort low
```

Output: `results/twostep_<model>.json` per model + `results/comparison.json`.

## Axes

| Side | Knobs (CLI flag → flows to) |
|------|-----------------------------|
| Retrieval | `--retrieve-reasoning-effort`, `--reranker-threshold`, `--max-context-chars` → `Retriever(...)` |
| Generation | `--model`, `--reasoning-effort`, `--max-completion-tokens` → `Generator(...)` |
| Judge | `--judge-model` → eval scorer (use GPT-4.1 — see stage 3 README) |

## Reference results

`results/eval_results_*.json` — outputs from a previous sweep on the
`feat/foundry-iq-migration` branch. See [`RESULTS.md`](RESULTS.md) for the
analysis (GPT-5 family beat GPT-4.1+BYOD on most quality metrics; GPT-5-mini
hit the best cost/quality point).

`results/eval_comparison.json` — schema-v2 comparison table from that sweep.
