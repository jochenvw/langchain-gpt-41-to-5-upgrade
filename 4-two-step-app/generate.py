"""Generate step — Azure AI Foundry Agent Service / Responses API.

Independent of retrieval. Takes a query + pre-retrieved context and returns
a response + token usage + latency.
Tune via constructor args:
    model, system_prompt_template, max_completion_tokens, reasoning_effort.
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path

# Allow `python 4-two-step-app/generate.py` to find shared/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import settings


DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable safety compliance assistant. "
    "Answer the user's question using ONLY direct extracts from the retrieved documents below.\n\n"
    "Guidelines:\n"
    "- Extract and quote relevant passages verbatim from the documents.\n"
    "- Do NOT paraphrase, summarize, or synthesize new text.\n"
    "- Combine the most relevant extracted passages to form the answer.\n"
    "- Cite source reference IDs inline (e.g., [0], [1]) after each extract.\n"
    "- Use bullet points to separate distinct extracted passages.\n"
    "- If no document passage directly answers the question, state that.\n\n"
    "--- Retrieved Documents ---\n{context}\n--- End Documents ---"
)


@dataclass
class GenerateResult:
    response: str
    latency_ms: float | None
    attempts: int
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None


class Generator:
    """Azure AI Foundry Agent Service Responses API client."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,   # low | medium | high (model-dependent)
    ):
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        self._credential = DefaultAzureCredential()
        self._project = AIProjectClient(
            endpoint=settings.foundry_endpoint,
            credential=self._credential,
        )
        self._openai = self._project.get_openai_client()

        self._model = model or settings.foundry_model_deployment
        self._system_prompt_template = system_prompt_template
        self._max_completion_tokens = max_completion_tokens
        self._reasoning_effort = reasoning_effort

    def generate(self, query: str, context: str) -> GenerateResult:
        """Call the Responses API with the retrieved context inlined.

        Retries once on content-filter refusals (transient GPT-5 behaviour).
        Returns a GenerateResult with timing + token usage.
        """
        if not context:
            return GenerateResult(
                response="I don't have enough information to answer that question.",
                latency_ms=None, attempts=0, model=self._model,
            )

        instructions = self._system_prompt_template.format(context=context)
        kwargs: dict = {
            "model": self._model,
            "instructions": instructions,
            "input": query,
        }
        if self._max_completion_tokens is not None:
            kwargs["max_output_tokens"] = self._max_completion_tokens
        if self._reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}

        attempts = 0
        start = _time.perf_counter()
        try:
            response = None
            text = ""
            for _ in range(2):
                attempts += 1
                response = self._openai.responses.create(**kwargs)
                text = response.output_text or ""
                if text and "cannot assist" not in text.lower():
                    break
            latency_ms = round((_time.perf_counter() - start) * 1000, 1)
            usage = response.usage if response else None
            return GenerateResult(
                response=text,
                latency_ms=latency_ms,
                attempts=attempts,
                model=self._model,
                prompt_tokens=usage.input_tokens if usage else 0,
                completion_tokens=usage.output_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )
        except Exception as exc:
            return GenerateResult(
                response="", latency_ms=None, attempts=attempts,
                model=self._model, error=str(exc),
            )


# ---------------------------------------------------------------------------
# CLI: quick smoke test (you must supply context inline or via stdin)
# ---------------------------------------------------------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Generator smoke test")
    parser.add_argument("query", nargs="?",
                        default="Summarize the safety guidelines.")
    parser.add_argument("--context", default="(no context provided)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None,
                        choices=[None, "low", "medium", "high"])
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    args = parser.parse_args()

    gen = Generator(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
    )
    result = gen.generate(args.query, args.context)
    print(f"Model      : {result.model}")
    print(f"Latency    : {result.latency_ms} ms")
    print(f"Tokens     : in={result.prompt_tokens} out={result.completion_tokens}")
    if result.error:
        print(f"ERROR      : {result.error}")
    print("--- response ---")
    print(result.response)


if __name__ == "__main__":
    _main()
