# 4 — Two-step app (NEW)

Drop-in replacement for `1-baseline-app/` that **decouples retrieval from
generation**. Two independently configurable steps:

```
retrieve.py   Foundry IQ KB retrieve API   (Azure AI Search agentic retrieval)
generate.py   Responses API via AIProjectClient  (any GPT-5 family deployment)
app.py        Composes them; CLI loop with citations + per-step latency
```

Why split: BYOD is a single opaque server-side call. Splitting lets us
swap models, tune retrieval, and measure each step independently — which
is what stage 5 sweeps over.

## Run

```bash
# Interactive chat
uv run python 4-two-step-app/app.py
uv run python 4-two-step-app/app.py -v             # full tracebacks

# Smoke-test each step alone
uv run python 4-two-step-app/retrieve.py "your query"
uv run python 4-two-step-app/generate.py "your query" --context "your context"
```

## Tunables

| File | Knobs |
|------|-------|
| `retrieve.py` (`Retriever(...)`) | `kb_name`, `knowledge_source_name`, `reranker_threshold`, `reasoning_effort` (minimal/low/medium/high), `max_context_chars`, `intent_kind` |
| `generate.py` (`Generator(...)`) | `model`, `system_prompt_template`, `max_completion_tokens`, `reasoning_effort` |

Stage 5's `run.py` exposes most of these as CLI flags.

## One-time setup: provision the Foundry IQ knowledge base

```bash
uv run python 4-two-step-app/setup_foundry_iq.py             # create KB + KS
uv run python 4-two-step-app/setup_foundry_iq.py --check     # verify
uv run python 4-two-step-app/setup_foundry_iq.py --teardown  # delete
```

Required env vars (see `.env.example`):
`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_INDEX`, `AZURE_OPENAI_ENDPOINT`,
`FOUNDRY_ENDPOINT`, `FOUNDRY_MODEL_DEPLOYMENT`.

## Architecture

```
user query
    │
    ▼
┌─────────────────┐     ┌──────────────────┐
│   Retriever     │────▶│    Generator     │────▶ response
│ (KB retrieve    │ ctx │ (Responses API,  │      + citations
│  + reranker)    │     │  no max_tokens)  │      + per-step latency
└─────────────────┘     └──────────────────┘
```

No `max_tokens` parameter is sent — uses `max_output_tokens` on the
Responses API instead, which works on GPT-5+.
