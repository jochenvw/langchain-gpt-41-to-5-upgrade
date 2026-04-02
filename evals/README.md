# Evaluations

Baseline evaluation framework for the LangChain Azure OpenAI app.

## Eval framework choice

This project uses the **Azure AI Evaluation SDK** (`azure-ai-evaluation`) for
built-in evaluators. However, **the eval framework is pluggable** — you can
bring your own:

| Framework | When to use |
|-----------|-------------|
| **Azure AI Evaluation SDK** (current) | Built-in RAG evaluators, Foundry portal integration |
| **RAGAS** | Popular open-source RAG evaluation with academic metrics |
| **DeepEval** | pytest-style LLM evals with CI-friendly assertions |
| **Custom scripts** | Full control, no extra dependencies |

The JSONL data files in `evals/data/` are framework-agnostic — any eval tool
can read them. Only the eval scripts (`eval_chat.py`, `eval_byod.py`) are
SDK-specific.

## Quick start

```bash
# Install dependencies (from project root)
uv sync

# Run all evals (results saved locally as JSON)
uv run python -m evals.run_all

# Run a single suite
uv run python -m evals.run_all --suite chat
uv run python -m evals.run_all --suite byod
```

## Foundry portal integration (optional)

When configured, eval results are **automatically uploaded** to the Azure AI
Foundry portal where you get:
- Visual dashboards with score charts
- Run-to-run comparison (e.g. GPT 4.1 vs GPT 5)
- Per-query drill-down with reasoning explanations
- Stored history of all eval runs

To enable, add these to your `.env`:

```bash
FOUNDRY_SUBSCRIPTION_ID=your-subscription-id
FOUNDRY_RESOURCE_GROUP=your-resource-group
FOUNDRY_PROJECT_NAME=your-foundry-project
```

**Prerequisites for Foundry portal:**
1. An Azure AI Foundry project
2. A storage account connected to the project
3. `Storage Blob Data Owner` role on the storage account for your identity

Leave these blank to run evals **locally only** — everything still works,
results are saved to `eval_results_*.json` files.

## Suites

### Chat (`eval_chat.py`) — lightweight
3 general Q&A queries evaluated for:
- **Coherence** — logical consistency
- **Fluency** — language quality
- **Relevance** — response addresses the query
- **F1 Score** — token overlap with ground truth (NLP, no LLM needed)

### BYOD / RAG (`eval_byod.py`) — baseline
10 safety-domain queries evaluated for:
- **Groundedness** — is the answer supported by retrieved context?
- **Relevance** — does the response address the query?
- **Coherence** — logical consistency
- **Fluency** — language quality
- **Retrieval** — did the search index return useful documents?

## Test data

Edit the JSONL files in `evals/data/` to match your domain:
- `chat_test_data.jsonl` — query + ground_truth pairs
- `byod_test_data.jsonl` — queries for your search index (context filled at runtime)

Generate data automatically from your search index:
```bash
uv run python scripts/generate_byod_eval_data.py --dry-run     # preview
uv run python scripts/generate_byod_eval_data.py --sample-size 15
```

## CI

The GitHub Actions workflow (`.github/workflows/eval.yml`) runs on PRs that
touch `app.py`, `config.py`, or `evals/**`. Configure these repo secrets:

| Secret | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | API key (CI uses key auth, not Entra) |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint |
| `AZURE_SEARCH_INDEX` | Search index name |
| `AZURE_SEARCH_API_KEY` | Search API key |

Optionally add `FOUNDRY_SUBSCRIPTION_ID`, `FOUNDRY_RESOURCE_GROUP`, and
`FOUNDRY_PROJECT_NAME` to push CI eval results to the Foundry portal.

You can also trigger evals manually via **Actions → Run LLM Evaluations → Run workflow**.
