# 1 — Baseline app (GPT-4.1 + BYOD)

The current state. **This is what we're replacing.**

LangChain `AzureChatOpenAI` → Azure OpenAI → On Your Data (BYOD) →
Azure AI Search index. Single API call.

## Run

```bash
uv run python 1-baseline-app/app.py --mode chat     # connectivity test
uv run python 1-baseline-app/app.py --mode direct   # raw OpenAI SDK
uv run python 1-baseline-app/app.py --mode byod     # GPT-4.1 + BYOD (works)
```

## Reproduce the GPT-5 break

Set `AZURE_OPENAI_DEPLOYMENT=gpt-5` in `.env`, then:

```bash
uv run python 1-baseline-app/app.py --mode byod
# → 400: 'max_tokens' is not supported with this model.
#       Use 'max_completion_tokens' instead.
```

The error originates inside Azure's BYOD pipeline. Not fixable client-side.
See [`../docs/findings.md`](../docs/findings.md).
