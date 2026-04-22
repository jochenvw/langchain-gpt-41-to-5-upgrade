# Evaluation Results — GPT-5 Family Model Comparison

## Architecture

This comparison uses a **two-step RAG pipeline** (stage 4) that decouples retrieval from generation:

```
Step 1: KB Retrieve  →  Azure AI Search knowledge base (semantic search, minimal reasoning)
Step 2: Generate     →  Azure OpenAI chat completions (per-model)
```

Retrieval is **cached and model-agnostic** — the same context is reused across all generation models so quality comparisons are apples-to-apples.

## Results (10 safety-domain queries)

### Latency & Quality

| Model | E2E | Retrieve | Generate | Groundedness | Relevance | Coherence | Fluency | Retrieval | Tokens |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|---:|
| **gpt-5.4-mini** | **6,925ms** | 929ms | 5,996ms | **5.00** | 4.30 | 4.30 | **4.00** | 4.40 | 25,446 |
| gpt-5.4-nano | 10,312ms | 929ms | 9,383ms | 4.70 | 4.30 | 4.20 | 4.00 | 4.30 | 26,389 |
| gpt-5.4-2 | 10,425ms | 929ms | 9,496ms | **5.00** | **4.80** | **4.60** | 4.00 | 4.40 | 26,944 |
| gpt-5.4 | 11,948ms | 929ms | 11,019ms | **5.00** | **4.80** | 4.20 | 3.90 | **4.50** | 26,558 |
| gpt-5.1 | 12,734ms | 929ms | 11,805ms | **5.00** | 4.43 | 4.29 | 4.00 | 4.29 | 29,597 |
| gpt-5-mini | ✗ | — | — | — | — | — | — | — | — |

> **gpt-5-mini** failed: only supports `temperature=1` (no temperature control).

### Key Findings

1. **Retrieval is fast and constant** — ~929ms average across all queries, model-agnostic
2. **Generation dominates latency** — accounts for 85–93% of end-to-end time
3. **gpt-5.4-mini is the clear speed winner** — 6.9s E2E with perfect groundedness (5.0)
4. **gpt-5.4-2 has the best overall quality** — 5.0 groundedness + highest relevance (4.8) and coherence (4.6)
5. **All models achieve perfect or near-perfect groundedness** (4.7–5.0) — the KB retrieval is doing its job

## Recommendation

| Priority | Model | Why |
|---|---|---|
| **★ Best overall** | `gpt-5.4-mini` | Fastest (6.9s), perfect groundedness, fewest tokens |
| Runner-up (quality) | `gpt-5.4-2` | Best relevance + coherence at ~50% more latency |
| Avoid | `gpt-5-mini` | Doesn't support temperature control |
| Avoid | `gpt-5.1` | Slowest, no quality advantage |

## Notes

- **BYOD baseline could not run** — the deployment is now GPT-5, which rejects `max_tokens` (requires `max_completion_tokens`). This is the exact BYOD bug that motivated the migration.
- **Judge model**: gpt-4.1 (via Azure AI Evaluation SDK)
- **Retrieval**: Azure AI Search KB with `minimal` reasoning effort (bypasses internal model call blocked by Azure Policy)
- Quality scores are on a 1–5 scale (higher is better)
- Token counts are generation-only (retrieve tokens are internal to the search service)

## How to Reproduce

```bash
# Run full comparison (all models)
uv run python 5-experiments/run_comparison.py

# Run specific models only
uv run python 5-experiments/run_comparison.py --models gpt-5.4-mini gpt-5.4-2

# Skip BYOD baseline
uv run python 5-experiments/run_comparison.py --skip-byod
```
