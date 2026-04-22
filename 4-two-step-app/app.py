"""Two-step app — GPT-5 RAG via Foundry IQ KB retrieve + Responses API.

Drop-in replacement for `1-baseline-app/app.py --mode byod`.
Composes `retrieve.py` (KB call) + `generate.py` (Responses API).

Each step is independently configurable — see retrieve.py / generate.py
for tunable parameters. This file just wires them together for an
interactive chat loop and an evaluable batch interface.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.config import settings

from retrieve import Retriever, RetrieveResult
from generate import Generator, GenerateResult


@dataclass
class TwoStepResult:
    response: str
    context: str
    citations: list[dict]
    retrieve_latency_ms: float
    generate_latency_ms: float | None
    generate_attempts: int
    generate_model: str
    generate_prompt_tokens: int = 0
    generate_completion_tokens: int = 0
    generate_total_tokens: int = 0
    error: str | None = None
    retrieve_backend: str = "foundry-kb-retrieve"

    def as_dict(self) -> dict:
        return asdict(self)


class TwoStepSession:
    """Holds long-lived Retriever + Generator clients for batch use (evals)."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()

    def query(self, user_query: str) -> TwoStepResult:
        r: RetrieveResult = self.retriever.retrieve(user_query)
        g: GenerateResult = self.generator.generate(user_query, r.context)
        return TwoStepResult(
            response=g.response,
            context=r.context,
            citations=r.citations,
            retrieve_latency_ms=r.latency_ms,
            generate_latency_ms=g.latency_ms,
            generate_attempts=g.attempts,
            generate_model=g.model,
            generate_prompt_tokens=g.prompt_tokens,
            generate_completion_tokens=g.completion_tokens,
            generate_total_tokens=g.total_tokens,
            error=g.error,
        )


def query_two_step(user_query: str) -> TwoStepResult:
    """Single-shot helper that creates a transient session."""
    return TwoStepSession().query(user_query)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def run_chat(verbose: bool = False) -> None:
    print("\n=== Two-step app — Foundry IQ KB retrieve + Responses API ===")
    print(f"Project : {settings.foundry_endpoint}")
    print(f"Model   : {settings.foundry_model_deployment}")
    print(f"KB      : {os.environ.get('FOUNDRY_KNOWLEDGE_BASE_NAME', 'safety-knowledge-base')}")
    print("Type 'quit' to exit.\n")

    session = TwoStepSession()
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            return

        try:
            result = session.query(user_input)
            if result.error:
                print(f"\n[ERROR] {result.error}\n")
                continue

            print(f"\nAssistant: {result.response}\n")
            print(
                f"  retrieve={result.retrieve_latency_ms}ms  "
                f"generate={result.generate_latency_ms}ms  "
                f"tokens={result.generate_prompt_tokens}+{result.generate_completion_tokens}\n"
            )

            if result.citations:
                print("--- Citations ---")
                for i, c in enumerate(result.citations, 1):
                    print(f"  [{i}] {c.get('title', 'N/A')}  "
                          f"(score={c.get('score')})")
                print()
        except Exception as exc:
            if verbose:
                traceback.print_exc()
            else:
                print(f"\n[ERROR] {type(exc).__name__}: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-step app — Foundry IQ KB retrieve + Responses API"
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show full tracebacks on errors")
    args = parser.parse_args()
    run_chat(verbose=args.verbose)


if __name__ == "__main__":
    main()
