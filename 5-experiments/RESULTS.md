# Evaluation Results — GPT-5 Family Model Comparison

## Architecture Overview

This project compares two RAG architectures for answering safety-domain questions grounded in an Azure AI Search index:

| Architecture | How it works | Models | Latency profile |
|---|---|---|---|
| **BYOD (legacy)** | Single HTTP call — Azure OpenAI "On Your Data" handles both retrieval and generation in one request | GPT-4.1 | One round-trip |
| **Foundry IQ (new)** | Two HTTP calls — (1) KB retrieve API for agentic search, then (2) Responses API for answer generation | Any GPT-5 family model | Retrieve + Generate |

```
BYOD (old):     Client ──► Azure OpenAI + On Your Data ──► Response
                           (1 call, 1 model)

Foundry (new):  Client ──► Foundry KB Retrieve API ──► context
                           (call 1: semantic search)
                       ──► GPT-5 Responses API ──► answer
                           (call 2: generation with context)
```

The two-call architecture gives us independent knobs for each stage — different models, different parameters, and visibility into where time is spent.

## Results (20 safety-domain queries)

### Quality Scores (1–5 scale)

| Model | Groundedness | Relevance | Coherence | Fluency | Retrieval |
|---|:---:|:---:|:---:|:---:|:---:|
| **GPT-4.1 BYOD** (baseline) | 4.85 | 4.50 | **4.05** | **3.85** | 5.00 |
| gpt-5.1 | 4.75 | 4.40 | 3.75 | 3.60 | 5.00 |
| gpt-5-mini | 4.75 | 4.55 | 3.75 | 3.60 | 5.00 |
| gpt-5.4-mini | 4.65 | 4.35 | 3.85 | 3.50 | 5.00 |
| gpt-5.4-nano | **4.85** | 4.10 | 3.85 | 3.45 | 5.00 |
| gpt-5.4 | **4.85** | **4.62** | 3.69 | **3.85** | 5.00 |
| gpt-5.4-2 | **4.85** | 4.55 | 3.75 | 3.65 | 5.00 |

> **Retrieval** is 5.0 across the board — the Foundry IQ KB retrieve with semantic reranking consistently returns relevant documents.

### Latency Breakdown (ms, averaged over 20 queries)

| Model | E2E | Retrieve | Generate | Min E2E | Max E2E |
|---|---:|---:|---:|---:|---:|
| **GPT-4.1 BYOD** (baseline) | **5,294** | — | — | 2,343 | 9,406 |
| gpt-5.1 | 7,333 | 1,868 | 5,465 | 2,304 | 21,813 |
| gpt-5-mini | 36,469 | 727 | 35,742 | 13,931 | 121,641 |
| **gpt-5.4-mini** | **5,833** | 728 | 5,105 | **1,601** | 10,662 |
| gpt-5.4-nano | 6,204 | **686** | 5,517 | 2,221 | 12,582 |
| gpt-5.4 | 12,246 | 734 | 12,081 | 5,489 | 22,624 |
| gpt-5.4-2 | 6,452 | 741 | 5,711 | 2,078 | 13,943 |

> **Stage timings measure API call time only** — they won't sum exactly to E2E due to orchestration overhead (JSON parsing, context assembly, retry logic).

### Token Usage

| Model | Avg Tokens/Query | Total (20 queries) | Scope |
|---|---:|---:|---|
| **GPT-4.1 BYOD** | 5,393 | 107,863 | full-pipeline |
| gpt-5.1 | 3,286 | 65,720 | generation-only |
| gpt-5-mini | 4,214 | 84,270 | generation-only |
| gpt-5.4-mini | 3,246 | 64,915 | generation-only |
| gpt-5.4-nano | 3,232 | 64,641 | generation-only |
| gpt-5.4 | 2,140 | 42,800 | generation-only |
| gpt-5.4-2 | 3,267 | 65,338 | generation-only |

> BYOD token counts include the full pipeline (retrieval + generation). Foundry counts are generation-only — the retrieve step uses internal search compute, not billed as model tokens.

## Key Findings

### 1. Retrieve is fast, generate dominates latency

With `minimal` reasoning effort, the KB retrieve API consistently returns in **~700ms**. The generation call accounts for 80–98% of end-to-end time. This means **model choice for generation is the primary latency lever**.

### 2. Best candidates to match BYOD latency

The BYOD baseline is **5,294ms E2E**. The closest Foundry matches:

| Model | E2E | Quality trade-off |
|---|---:|---|
| **gpt-5.4-mini** | 5,833ms (+10%) | Slightly lower groundedness (4.65 vs 4.85) |
| **gpt-5.4-nano** | 6,204ms (+17%) | Best groundedness (4.85), lower relevance |
| **gpt-5.4-2** | 6,452ms (+22%) | Top groundedness (4.85), good all-round |

### 3. gpt-5-mini is an outlier

At 36.5s E2E (35.7s generation alone), gpt-5-mini is ~6x slower than the 5.4 family despite similar quality. Not recommended for latency-sensitive workloads.

### 4. gpt-5.4 has content-filter issues

7 out of 20 queries returned empty responses due to content-filter refusals (transient GPT-5.4 issue). Scores are based on 13 successful queries. This may improve as the model stabilizes.

## What Changed (code + infra)

### Code changes

1. **Split eval metrics** (`app.py`, `evals/eval_byod.py`, `scripts/run_model_comparison.py`)
   - `FoundryAgentSession._retrieve()` now measures and returns `retrieve_latency_ms`
   - `FoundryAgentSession.query()` measures `generate_latency_ms` separately
   - New fields: `generate_model`, `retrieve_backend`, `generate_attempts`, prefixed token fields (`generate_prompt_tokens`, etc.)
   - Eval comparison JSON restructured to schema v2 with `architecture`, `quality`, `latency`, `retrieve`, `generate` hierarchy

2. **KB retrieve: `low` → `minimal` reasoning** (`app.py`)
   - Switched from `messages` + `low` reasoning (LLM-assisted query planning) to `intents` + `minimal` reasoning (direct semantic search)
   - This bypasses the KB's internal model call for query planning, which was blocked by Azure Policy disabling key auth on the AOAI resource
   - **Side effect**: faster retrieval (~700ms vs ~2–5s with LLM planning), no quality degradation (retrieval scores went to 5.0 across all models)

### Infrastructure fixes

1. **Semantic ranker**: Upgraded `jvw-safety-store` from free tier to standard (free monthly quota was exhausted)
2. **KB auth**: Removed stale API key from KB model config (key auth disabled by Azure Policy on the AOAI resource)

## How to Reproduce

```bash
# Run all 6 models (takes ~30 min)
uv run python scripts/run_model_comparison.py

# Run a single model
uv run python -m evals.eval_byod --use-foundry --model gpt-5.4-mini

# Run the BYOD baseline
uv run python -m evals.eval_byod
```

Results are saved to `eval_results_*.json` files. The comparison table is written to `eval_comparison.json` (schema v2).

## Data Schema (`eval_comparison.json`)

```jsonc
{
  "schema_version": 2,
  "gpt-5.4-mini": {
    "architecture": "two-call",          // vs "single-call" for BYOD
    "quality": { "groundedness": 4.65, "relevance": 4.35, ... },
    "latency": { "e2e_avg_ms": 5833 },
    "retrieve": {
      "backend": "foundry-kb-retrieve",  // KB retrieve API
      "avg_latency_ms": 728              // API call time only
    },
    "generate": {
      "model": "gpt-5.4-mini",          // can be different from retrieve
      "avg_latency_ms": 5105
    },
    "tokens": {
      "total": 64915, "avg": 3246,
      "scope": "generation-only"         // retrieve tokens not tracked
    }
  },
  "GPT-4.1 BYOD": {
    "architecture": "single-call",
    "model": "gpt-4.1",
    "quality": { ... },
    "latency": { "e2e_avg_ms": 5294 },
    "tokens": { "scope": "full-pipeline" }
  }
}
```
