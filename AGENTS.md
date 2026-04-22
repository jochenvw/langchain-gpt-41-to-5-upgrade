# Repository conventions for AI coding assistants

This file is read by GitHub Copilot and other LLM coding assistants. Follow
the rules below when generating or editing code in this repository.

## Architecture decisions (ADRs)

All accepted ADRs in [`docs/adr/`](docs/adr/) are binding. Read the index at
[`docs/adr/README.md`](docs/adr/README.md). The current set:

- **[ADR-0001](docs/adr/0001-prompts-as-markdown-files.md) — Prompts live in
  markdown files, not in source code.** All system prompts and
  answer-generation prompts MUST live under `prompts/` and be loaded via
  `shared.prompts.load(name)`. Do NOT inline multi-line prompt strings in
  `.py` source. When changing a prompt, edit the markdown file; when adding
  one, create a new markdown file and update `prompts/README.md`.

## Repository layout

The repo is organised into numbered migration stages:

```
1-baseline-app/      LangChain GPT-4.1 + BYOD baseline
2-eval-data/         Synthetic eval-set generation (ARES-validated)
3-baseline-eval/     Score the baseline app against the eval set
4-two-step-app/      retrieve.py + generate.py replacement for BYOD
5-experiments/       Sweep models / retrieval / generation knobs
shared/              Shared config + prompt loader
prompts/             All prompts (see ADR-0001)
docs/                Findings, playbook, ADRs
```

Each stage has its own README; the top-level [README](README.md) links them
in order.

## Tooling

- Package manager: `uv` (root `pyproject.toml`, single venv shared by all stages)
- Run anything: `uv run python <stage>/<script>.py --help`
- Lint/test: there is no enforced linter; keep changes minimal and verified
  via `--help` smoke tests + targeted `uv run` invocations.
