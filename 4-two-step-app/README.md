# 4 — Two-step app (NEW)

Drop-in replacement for `1-baseline-app/` that decouples retrieval from
generation. Two independently configurable stages:

```
retrieve.py    Azure AI Search call (top_k, semantic, reranker, filters, ...)
generate.py    Model call (model, reasoning_effort, max_completion_tokens, ...)
app.py         Composes them; CLI mirrors 1-baseline-app/app.py
```

This unlocks model choice (any GPT-5 family / o-series / non-Azure) and
matrix experimentation (stage 5).

## Status

> **TODO** — populated from `feat/foundry-iq-migration` branch in a follow-up
> PR. That branch already implements KB retrieve → AIProjectClient Responses
> API generation; it just needs to be split into `retrieve.py` + `generate.py`
> here.
