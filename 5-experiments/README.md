# 5 — Experiments

Sweep retrieve × generate configurations to find the cheapest setup that
matches or beats the baseline numbers from stage 3.

## Axes

| Side | Knobs |
|------|-------|
| Retrieval (stage 4 `retrieve.py`) | `top_k`, semantic on/off, reranker threshold, filters, output mode (extractive/abstractive), context cap |
| Generation (stage 4 `generate.py`) | model (gpt-5, gpt-5-mini, gpt-5.1, gpt-5.4-{mini,nano,full}), `reasoning_effort` (low/medium/high), `max_completion_tokens` |

## Status

> **TODO** — populated from `feat/foundry-iq-migration` branch. That branch
> already has `scripts/run_model_comparison.py` (8-model sweep) and an
> `EVAL_RESULTS.md` with full comparison; both need to land here.
