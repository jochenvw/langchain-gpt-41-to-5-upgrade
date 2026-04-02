# LangChain GPT 4.1 → GPT 5 Upgrade Investigation

## Purpose

Investigate the work required to upgrade a LangChain agent that uses Bring-Your-Own-Data (BYOD) from GPT 4.1 to GPT 5, as GPT 4.1 will be deprecated.

## Context

A customer reported that BYOD (On Your Data) breaks when the model is upgraded from GPT 4.1 to GPT 5. This repo reproduces the issue and documents the upgrade path.

## Architecture Overview

- **LangChain app** connecting to Azure OpenAI via APIM gateway
- **GPT 4.1** model deployment (target: upgrade to GPT 5)
- **Azure AI Search** for BYOD (On Your Data) with index `safety-source-index`
- **Endpoint**: Azure OpenAI through APIM gateway

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Python** | ≥ 3.12 | [python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.12` |
| **uv** | ≥ 0.4 | `pip install uv` or `winget install astral-sh.uv` — see [docs.astral.sh](https://docs.astral.sh/uv/getting-started/installation/) |

You also need access to:
- An **Azure OpenAI** resource with a GPT 4.1 (or GPT 5) deployment
- An **Azure AI Search** service with a populated index (for BYOD mode)
- Either an **API key** or **Azure Entra ID** credentials for authentication

## Getting started

```bash
# 1. Clone the repo
git clone <repo-url>
cd langchain-gpt-41-to-5-upgrade

# 2. Create a virtual environment and install all dependencies
uv sync

# 3. Copy the environment template and fill in your credentials
cp .env.example .env
#    Edit .env with your Azure OpenAI and Azure AI Search values
```

### Environment variables (.env)

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI (or APIM) endpoint | `https://your-resource.openai.azure.com/openai` |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name | `gpt-4.1` |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-06-01` |
| `AZURE_OPENAI_AUTH_TYPE` | `entra` (Azure AD) or `key` | `entra` |
| `AZURE_OPENAI_API_KEY` | API key (only needed if auth_type=key) | |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint | `https://your-search.search.windows.net` |
| `AZURE_SEARCH_INDEX` | Search index name | `safety-source-index` |
| `AZURE_SEARCH_AUTH_TYPE` | `token`, `key`, or `rbac` | `token` |
| `AZURE_SEARCH_API_KEY` | Search API key (only needed if auth_type=key) | |
| `FOUNDRY_SUBSCRIPTION_ID` | *(optional)* Azure subscription for Foundry portal | |
| `FOUNDRY_RESOURCE_GROUP` | *(optional)* Resource group for Foundry project | |
| `FOUNDRY_PROJECT_NAME` | *(optional)* Foundry project name for eval dashboards | |

## Running the app

```bash
# Simple chat (no retrieval) — good for testing connectivity
uv run python app.py --mode chat

# Direct OpenAI SDK test — bypasses LangChain
uv run python app.py --mode direct

# BYOD / On Your Data — chat grounded in Azure AI Search
uv run python app.py --mode byod

# Add -v for full tracebacks on errors
uv run python app.py --mode byod -v
```

## Running evaluations

```bash
# Run all eval suites (chat + BYOD)
uv run python -m evals.run_all

# Run just the chat eval (small, 3 queries)
uv run python -m evals.run_all --suite chat

# Run just the BYOD/RAG eval (10 queries, baseline quality)
uv run python -m evals.run_all --suite byod
```

Results are saved to `eval_results_chat.json` and `eval_results_byod.json`.
See [evals/README.md](evals/README.md) for details on evaluators and test data.

## Known Issues

- BYOD (On Your Data) is **deprecated** and does not support GPT 5
- `temperature=0.7` is rejected by GPT 5 (only default `1` is accepted)
- `max_tokens` must be replaced with `max_completion_tokens` for GPT 5
- See [UPGRADE_NOTES.md](UPGRADE_NOTES.md) for detailed findings

## Documentation

| Document | Description |
|----------|-------------|
| [UPGRADE_NOTES.md](UPGRADE_NOTES.md) | Investigation findings — what breaks and why |
| [MIGRATION_PLAYBOOK.md](MIGRATION_PLAYBOOK.md) | Step-by-step migration guide (GPT 4.1 + BYOD → GPT 5 + Foundry IQ) |
| [evals/README.md](evals/README.md) | Evaluation framework docs |
