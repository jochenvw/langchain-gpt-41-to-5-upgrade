# Evaluations — Azure AI Evaluation SDK

Baseline evaluation framework for the LangChain Azure OpenAI app.

## Quick start

```bash
# Install eval dependencies (from project root)
uv sync

# Run all evals
python -m evals.run_all

# Run a single suite
python -m evals.run_all --suite chat
python -m evals.run_all --suite byod
```

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

For BYOD, populate `ground_truth` in the JSONL once you have verified correct answers
from your safety index to enable F1/similarity scoring.

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

You can also trigger evals manually via **Actions → Run LLM Evaluations → Run workflow**.
