# LangChain GPT-4.1 + BYOD → GPT-5+ migration

A fact-driven migration path. Numbered folders **are** the workflow.

```
1-baseline-app/      Current app (GPT-4.1 + BYOD via On Your Data)
2-eval-data/         Generate ARES-validated synthetic eval set
3-baseline-eval/     Score baseline app → numbers to match/beat
4-two-step-app/      NEW: retrieve → generate (drop-in BYOD replacement)
5-experiments/       Sweep AI Search × model/reasoning/token settings
shared/              Code shared across stages (config, prompt loader)
prompts/             All system + answer-generation prompts (see ADR-0001)
docs/                Findings, playbook, ADRs
kb-research/         Orthogonal POC: how to build the KB itself
```

## Why

BYOD (Azure OpenAI On Your Data) is **deprecated** and breaks on GPT-5
(server-side `max_tokens`). This repo documents how to migrate by
**decoupling retrieval from generation** so you can choose any model.

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.12 |
| uv | ≥ 0.4 (`pip install uv`) |
| Azure OpenAI | GPT-4.1 (baseline) and/or GPT-5 deployment |
| Azure AI Search | Index populated with your data |

```bash
uv sync
cp .env.example .env   # fill in your endpoints/keys
```

## The path (run stages in order)

```bash
# 1. Reproduce the baseline locally (and watch BYOD break on GPT-5)
uv run python 1-baseline-app/app.py --mode byod

# 2. Generate ARES-validated synthetic eval set from your search index
uv run python 2-eval-data/generate.py

# 3. Score the baseline app to establish target numbers
uv run python 3-baseline-eval/run.py

# 4. Two-step replacement (retrieve → generate, configurable per step)
uv run python 4-two-step-app/setup_foundry_iq.py    # one-time KB provisioning
uv run python 4-two-step-app/app.py                  # interactive chat

# 5. Sweep model + retrieval settings to match/beat baseline
uv run python 5-experiments/run.py --model gpt-5-mini --reasoning-effort low
uv run python 5-experiments/sweep.py                 # full matrix
```

## Findings (TL;DR)

- `temperature` other than `1` → rejected by GPT-5
- `max_tokens` → must be `max_completion_tokens` for GPT-5
- BYOD pipeline itself sends `max_tokens` server-side → not fixable client-side
- See [`docs/findings.md`](docs/findings.md) and [`docs/playbook.md`](docs/playbook.md)
- Architecture decisions: [`docs/adr/`](docs/adr/) — start with [ADR-0001 (prompts as markdown)](docs/adr/0001-prompts-as-markdown-files.md)
- Conventions for AI coding assistants: [`AGENTS.md`](AGENTS.md)

## Environment variables

See [`.env.example`](.env.example). Required:
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_AUTH_TYPE`,
`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_INDEX`.

