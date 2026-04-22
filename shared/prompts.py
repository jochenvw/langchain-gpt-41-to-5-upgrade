"""Prompt loader.

Reads markdown files from `prompts/` (repo root). The file body — everything
after the first line consisting solely of `---` — is returned verbatim, with
surrounding whitespace stripped.

See ADR-0001 (docs/adr/0001-prompts-as-markdown-files.md).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the prompt body for `prompts/<name>.md`.

    `name` is the filename without the `.md` extension. Variables in the
    prompt use Python `str.format` syntax — call `.format(...)` at the use
    site.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt '{name}' not found at {path}. "
            "All prompts must live in the prompts/ directory — see ADR-0001."
        )
    text = path.read_text(encoding="utf-8")
    # Body = everything after the first standalone '---' separator line.
    parts = text.split("\n---\n", 1)
    body = parts[1] if len(parts) == 2 else text
    return body.strip() + "\n"
