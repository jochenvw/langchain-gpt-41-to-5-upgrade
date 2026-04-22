# ADR-0001: Prompts live in markdown files, not in source code

**Status**: Accepted
**Date**: 2026-04-22
**Deciders**: Repo maintainer

## Context

This repo is a fact-driven migration playbook (GPT-4.1 + BYOD → GPT-5+ via a
two-step retrieve+generate replacement). Prompts are a first-class lever
in that migration: every measured quality delta in `5-experiments/` is a
function of (model, retrieval settings, **prompt**).

When prompts are inlined as Python string literals scattered across several
files, three things break:

1. **Diffability.** Reviewing a prompt change in a code diff is noisy —
   surrounding indentation, escaping, and concatenation hide the actual
   semantic edit.
2. **Reuse.** The same "you are a helpful assistant" string was duplicated
   in 4 files. Drift was inevitable.
3. **LLM-assisted editing.** When Copilot (or any agent) edits prompts,
   it tends to mix prompt edits with code edits in the same hunk and to
   re-flow string concatenation, making intent hard to verify.

## Decision

**All system prompts and answer-generation prompts MUST live as markdown
files under `prompts/` at the repo root.** Code loads them via the
`shared.prompts.load(name)` helper.

Inlining a multi-line prompt as a string literal in `.py` source is
prohibited.

### File format

```
# Prompt: <name>

**Used by**: <files>
**Purpose**: <one line>
**Variables**: <list, or "none">

---

<prompt body>
```

Everything after the first standalone `---` line is the prompt body;
the loader returns it verbatim. Variables use Python `str.format` syntax
(`{name}`); literal braces are doubled (`{{`, `}}`).

### Loader contract

```python
from shared.prompts import load

prompt = load("two_step_extractive").format(context=ctx)
```

`load` raises `FileNotFoundError` on a missing prompt — fail-loud is
intentional so missing prompts surface immediately in dev/CI rather than
falling back to a silent default.

### What counts as a "prompt"

In scope:
- System prompts / system instructions
- Answer-generation prompts (multi-line user templates)
- Eval-data generation prompts

Out of scope (may stay in code):
- Single-line user message templates that are essentially just string
  interpolation (e.g. `f"Document A:\n\n{excerpt_a}"`).
- Per-call dynamic content (citations, retrieved passages) — that is data,
  not a prompt.

When in doubt: if the string is more than ~3 lines or shapes model behaviour,
it is a prompt — externalise it.

## Consequences

**Positive**
- Prompt changes show up as standalone markdown diffs.
- All prompts are discoverable in one directory with a single index
  (`prompts/README.md`).
- LLM-assisted editing tools (Copilot, agents) treat prompts as documents,
  not as code, reducing accidental code-side regressions.
- Easy to A/B prompts in `5-experiments/` — pass a different `name` to the
  loader.

**Negative**
- One extra file per prompt + one indirection (`load("name")`).
- Tests that snapshot exact prompt text must read from the same loader to
  stay in sync.

## Compliance — for human and LLM contributors

When adding or modifying any prompt:

1. The prompt **must** live in `prompts/<snake_case_name>.md` and follow
   the file format above.
2. Code must access it via `shared.prompts.load("<name>")`.
3. Update the index table in `prompts/README.md`.
4. Do not inline the prompt body in `.py` source — even temporarily.

LLM coding assistants (Copilot, Claude, etc.) operating in this repo MUST
follow this ADR. If a task requires a prompt change, edit the markdown
file; if it requires a new prompt, create a new markdown file. Never
re-inline a prompt back into Python source.

## References

- `prompts/README.md` — index + format spec
- `shared/prompts.py` — loader
