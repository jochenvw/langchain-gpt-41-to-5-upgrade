# GPT 4.1 → GPT 5 Migration Playbook

> **Audience**: An LLM agent (or developer) executing this migration step-by-step.
> Each step is self-contained with explicit inputs, outputs, and verification criteria.

## Prerequisites

| Requirement | Version | Why |
|-------------|---------|-----|
| **Python** | **≥ 3.12** | The `azure-ai-evaluation` SDK has a validator bug (`isinstance(v, typing.Any)` raises `TypeError`) on Python 3.11 and below. The `credential` field in `AzureOpenAIModelConfiguration` triggers this. **3.12+ is required.** |
| **uv** | ≥ 0.4 | Package manager — all commands use `uv sync` / `uv run`. Install: `pip install uv` or [docs.astral.sh](https://docs.astral.sh/uv/getting-started/installation/) |
| **Azure OpenAI** | — | A resource with GPT 4.1 deployment (for baseline) and GPT 5 (for migration target) |
| **Azure AI Search** | — | A search service with a populated index |
| **Auth** | Entra ID or API key | Entra ID (`DefaultAzureCredential`) is the default; API key also supported |

> ⚠️ **Known SDK issue**: `azure-ai-evaluation` cannot pass `credential` (DefaultAzureCredential)
> directly to evaluators due to a `typing.Any` validation bug. The workaround in
> `evals/eval_config.py` acquires a bearer token and passes it as `api_key` instead.
> This works but tokens expire after ~1 hour. For long eval runs, re-run the script.

## Background

Azure OpenAI On Your Data (BYOD) is **deprecated** and does not support GPT 5.
The migration path is:

1. Baseline current quality with evals on GPT 4.1
2. Generate real evaluation data from the existing BYOD pipeline
3. Create a standalone knowledge base in Azure AI Search (decoupled from On Your Data)
4. Replace BYOD with Foundry Agent Service + Foundry IQ for grounded answers
5. Apply GPT 5 client-side fixes (temperature, max_tokens) and switch the model
6. Re-run evals to confirm quality parity

## Repository structure

```
├── app.py                  # LangChain app — chat, BYOD, and direct SDK modes
├── config.py               # Loads .env, exposes AzureOpenAIConfig dataclass
├── requirements.txt        # Python deps (includes azure-ai-evaluation)
├── .env                    # Runtime config (endpoint, deployment, search index)
├── UPGRADE_NOTES.md        # Investigation findings (temperature, max_tokens, BYOD)
├── evals/
│   ├── eval_config.py      # Shared eval config — reads .env, builds model_config dict
│   ├── eval_chat.py        # Chat mode eval (3 queries, 4 evaluators)
│   ├── eval_byod.py        # BYOD/RAG eval (10 queries, 5 evaluators)
│   ├── run_all.py          # Runner — python -m evals.run_all --suite chat|byod|all
│   ├── data/
│   │   ├── chat_test_data.jsonl   # Chat eval dataset (query + ground_truth)
│   │   └── byod_test_data.jsonl   # BYOD eval dataset (query, context filled at runtime)
│   └── README.md
└── .github/workflows/eval.yml  # CI — runs evals on PRs, uploads artifacts
```

## Key technical details

### Current app architecture
- `app.py:build_llm()` creates an `AzureChatOpenAI` instance with `temperature=0.7`
- `app.py:get_byod_extra_body()` builds a `data_sources` payload for Azure AI Search
  - Search index: configured via `AZURE_SEARCH_INDEX` env var (currently `safety-source-index`)
  - Auth: `AZURE_SEARCH_AUTH_TYPE` — supports `key`, `rbac`, `token`
- `app.py:test_direct_openai()` uses the raw `openai.AzureOpenAI` SDK with `temperature=0.7`
- `config.py` strips `/openai` suffix from endpoint so the SDK constructs URLs correctly

### Current .env values
```
AZURE_OPENAI_ENDPOINT=https://energy-risk-demos.openai.azure.com/openai
AZURE_OPENAI_AUTH_TYPE=entra
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_DEPLOYMENT=gpt-5          # ← switch to gpt-4.1 for baseline
AZURE_SEARCH_ENDPOINT=https://jvw-safety-store.search.windows.net
AZURE_SEARCH_INDEX=safety-source-index
AZURE_SEARCH_AUTH_TYPE=token
```

### GPT 5 breaking changes (confirmed)
1. **`temperature`** — only default value `1` accepted. Code passes `0.7` → 400 error.
   - Affected: `app.py` line 35 (`build_llm`) and line 203 (`test_direct_openai`)
2. **`max_tokens`** — rejected, must use `max_completion_tokens` instead.
3. **On Your Data** — deprecated, not supported for GPT 5. The Azure pipeline itself sends
   `max_tokens` internally. No client-side fix exists.

### Eval framework
- Uses `azure-ai-evaluation` SDK (installed via `uv sync`)
- AI-assisted evaluators require an Azure OpenAI deployment to score (judge model)
- `eval_config.py:get_model_config()` returns the config dict evaluators need
- The BYOD eval (`eval_byod.py`) captures citations/context from response metadata
  so groundedness can be measured against retrieved documents

---

## Step 1: Scaffold evaluation framework ✅ DONE

**What was done:**
- Created `evals/` directory with Azure AI Evaluation SDK-based eval scripts
- Chat suite: 3 queries, evaluators = Coherence, Fluency, Relevance, F1Score
- BYOD suite: 10 safety-domain queries, evaluators = Groundedness, Relevance, Coherence, Fluency, Retrieval
- Runner script (`run_all.py`) with `--suite` flag
- GitHub Actions CI workflow on PRs

**Files created:**
- `evals/eval_config.py`, `evals/eval_chat.py`, `evals/eval_byod.py`
- `evals/run_all.py`, `evals/__init__.py`, `evals/README.md`
- `evals/data/chat_test_data.jsonl`, `evals/data/byod_test_data.jsonl`
- `.github/workflows/eval.yml`
- Updated `requirements.txt` to include `azure-ai-evaluation>=1.0.0`

**Verification:**
```bash
# Imports pass
python -c "from azure.ai.evaluation import GroundednessEvaluator, evaluate; print('OK')"
python -c "from evals.eval_config import get_model_config; print('OK')"
```

**Next action for an LLM agent:** Proceed to Step 2.

---

## Step 2: Generate real evaluation data from existing BYOD pipeline

> **Status**: NOT STARTED
> **Prerequisite**: Step 1 complete. `.env` must have `AZURE_OPENAI_DEPLOYMENT=gpt-4.1` (BYOD only works on 4.1).

**Goal:** Run the BYOD pipeline against the `safety-source-index` with a curated set of
queries and capture (query, response, context, ground_truth) tuples as the baseline dataset.

**Script:** `scripts/generate_byod_eval_data.py` — automates the entire flow:
1. Connects directly to Azure AI Search and samples documents
2. Uses GPT to synthesise realistic eval queries from document content
3. Runs each query through the BYOD pipeline to capture response + context
4. Writes a complete JSONL dataset to `evals/data/byod_test_data.jsonl`

**Instructions for an LLM agent:**

1. **Ensure `.env` has `AZURE_OPENAI_DEPLOYMENT=gpt-4.1`** so BYOD works.

2. **Dry run** to preview what documents are available:
   ```bash
   uv run python scripts/generate_byod_eval_data.py --dry-run --sample-size 15
   ```

3. **Generate the dataset** (queries + BYOD responses + context + ground truth):
   ```bash
   uv run python scripts/generate_byod_eval_data.py --sample-size 15 --queries-per-doc 2
   ```
   This produces ~30 eval queries. Use `--skip-byod` if you only want queries without
   running the BYOD pipeline (responses/context will be populated at eval time instead).

4. **Run the BYOD eval against this baseline** to get initial scores:
   ```bash
   uv run python -m evals.run_all --suite byod
   ```

5. **Save the baseline scores** — commit `eval_results_byod.json` or record the aggregate
   metrics (groundedness, relevance, retrieval) as the target to match after migration.

**Script options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--sample-size` | 10 | Number of documents to sample from the index |
| `--queries-per-doc` | 2 | Eval queries to generate per document |
| `--output` | `evals/data/byod_test_data.jsonl` | Output path |
| `--skip-byod` | off | Skip BYOD pipeline (only generate queries) |
| `--dry-run` | off | Preview sampled docs without calling GPT |

**Verification:**
- `byod_test_data.jsonl` exists with populated query, response, context, ground_truth fields
- `uv run python -m evals.run_all --suite byod` completes and prints aggregate scores
- Baseline scores are recorded for comparison

**Key implementation details:**
- The BYOD target function in `eval_byod.py` already extracts context from
  `result.response_metadata.context.citations[].content` — reuse this pattern
- For ground_truth: either have a human annotate, or use GPT 4.1 itself to generate
  reference answers and then human-verify a sample
- Store raw API responses for debugging if scores are unexpectedly low

---

## Step 3: Create standalone knowledge base in Azure AI Search

> **Status**: NOT STARTED
> **Prerequisite**: Step 2 complete (baseline established).

**Goal:** Ensure the Azure AI Search index (`safety-source-index` on `jvw-safety-store`)
is usable independently of the On Your Data pipeline — i.e., the app can query it
directly via the Azure AI Search SDK and pass results as context to the LLM.

**Instructions for an LLM agent:**

1. **Audit the existing search index** — use the Azure AI Search REST API or SDK to inspect:
   - Index schema (fields, types, searchable/filterable/retrievable attributes)
   - Document count and sample documents
   - Any skillsets or indexers attached
   ```
   GET https://jvw-safety-store.search.windows.net/indexes/safety-source-index?api-version=2024-07-01
   ```

2. **Build a direct search client** — create a module (e.g., `search_client.py`) that:
   - Uses `azure-search-documents` SDK to query the index directly
   - Accepts a query string, returns top-k results with content and metadata
   - Supports the same auth modes as the current app (token, key, rbac)

3. **Build a RAG chain** that replaces the BYOD pipeline:
   - Query Azure AI Search directly (not via On Your Data)
   - Format retrieved documents as context
   - Pass context + query to GPT model via LangChain
   - This decouples from the deprecated On Your Data feature

4. **Add `azure-search-documents` to `requirements.txt`**

5. **Run the BYOD eval suite** against the new RAG chain and compare to Step 2 baseline

**Verification:**
- Direct search queries return results matching what BYOD returned
- RAG chain produces grounded answers
- Eval scores are within acceptable range of Step 2 baseline

---

## Step 4: Switch to Foundry Agent Service + Foundry IQ

> **Status**: NOT STARTED
> **Prerequisite**: Step 3 complete (standalone RAG chain working).

**Goal:** Replace the custom RAG chain with Azure AI Foundry Agent Service and Foundry IQ,
which is Microsoft's recommended replacement for On Your Data.

**Instructions for an LLM agent:**

1. **Research Foundry IQ** — fetch the latest docs:
   - https://learn.microsoft.com/en-us/azure/ai-foundry/agents/concepts/what-is-foundry-iq
   - https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/foundry-iq-connect

2. **Provision Foundry resources** — create or configure:
   - Azure AI Foundry project
   - Connect the existing `safety-source-index` as a Foundry IQ knowledge base
   - Deploy GPT 5 model within the Foundry project

3. **Update the app** to use the Foundry Agent SDK instead of LangChain BYOD:
   - Replace `get_byod_extra_body()` with Foundry Agent invocation
   - Keep the LangChain chat path for non-RAG queries
   - Apply GPT 5 client-side fixes:
     - Remove `temperature=0.7` from `build_llm()` (line 35) and `test_direct_openai()` (line 203)
     - Use `max_completion_tokens` instead of `max_tokens` anywhere token limits are set

4. **Update .env** — set `AZURE_OPENAI_DEPLOYMENT=gpt-5` and add any Foundry-specific config

5. **Run full eval suite** against the Foundry-backed pipeline:
   ```bash
   python -m evals.run_all
   ```

6. **Compare scores** to Step 2 baseline — accept if within tolerance, investigate regressions

**Verification:**
- App runs in all modes (chat, byod-replacement, direct) without errors on GPT 5
- Eval scores meet or exceed Step 2 baseline
- No `temperature` or `max_tokens` errors
- CI workflow passes

---

## Success criteria

The migration is complete when:
1. All app modes work on GPT 5 without errors
2. BYOD functionality is replaced by Foundry IQ (no dependency on deprecated On Your Data)
3. Eval scores (groundedness, relevance, retrieval) are ≥ the GPT 4.1 BYOD baseline
4. CI pipeline runs evals on every PR
5. No references to deprecated On Your Data API remain in production code
