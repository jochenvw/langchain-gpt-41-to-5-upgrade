# Architecture Decision Records (ADRs)

Lightweight records of decisions that constrain the codebase. New
contributors (human or LLM) MUST respect the accepted ADRs.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-prompts-as-markdown-files.md) | Prompts live in markdown files, not in source code | Accepted |

## Format

ADRs use a minimal Michael Nygard-style template:

```
# ADR-NNNN: <title>

**Status**: Proposed | Accepted | Superseded by ADR-XXXX
**Date**: YYYY-MM-DD
**Deciders**: <who>

## Context
## Decision
## Consequences
```

## Adding an ADR

1. Pick the next number (`0002`, `0003`, ...).
2. Create `docs/adr/NNNN-kebab-title.md` using the template above.
3. Add a row to the table here.
4. Discuss in PR; flip status to **Accepted** on merge.
