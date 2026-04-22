# Prompts

All system / answer-generation prompts used anywhere in this repo live here, as
plain markdown. Code loads them via `shared.prompts.load(name)`.

This is mandated by [ADR-0001](../docs/adr/0001-prompts-as-markdown-files.md) —
**read it before adding or changing prompts**.

## Files

| File | Used by | Variables |
|------|---------|-----------|
| `helpful_assistant.md` | `1-baseline-app/app.py`, `3-baseline-eval/eval_chat.py`, `3-baseline-eval/eval_byod.py`, `2-eval-data/generate.py` (baseline replay) | — |
| `two_step_extractive.md` | `4-two-step-app/generate.py` | `{context}` |
| `eval_data_single_doc.md` | `2-eval-data/generate.py` (single-doc gen) | `{queries_per_doc}` |
| `eval_data_multi_doc.md` | `2-eval-data/generate.py` (multi-doc gen) | — |

## Format

```
# Prompt: <name>

**Used by**: <files>
**Purpose**: <one line>
**Variables**: <list, or "none">

---

<prompt body — everything after the first `---` line, trimmed>
```

The loader returns the body verbatim. Variables use Python `str.format` syntax
(`{name}`); double the braces for literal braces (`{{`, `}}`).

## Usage

```python
from shared.prompts import load

system_prompt = load("two_step_extractive").format(context=context_str)
```

## Adding a new prompt

1. Create `prompts/<snake_case_name>.md` following the format above.
2. Reference it via `load("<name>")` from the call site.
3. Add a row to the table above.
4. Do **not** inline a prompt string in Python code. See ADR-0001.
