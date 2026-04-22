# 3 — Baseline eval

Scores `1-baseline-app/` against `2-eval-data/`. Output **is the bar to beat**
for stage 4 and 5.

## Evaluators (azure-ai-evaluation SDK)

| Evaluator | Question |
|-----------|----------|
| Groundedness | Is the answer grounded in retrieved context? |
| Relevance | Does the response address the query? |
| Coherence | Is the response logically consistent? |
| Fluency | Is the response well-written? |
| Retrieval | Did the retriever return useful documents? |

Plus performance metrics: response time, time-to-first-token, response length.

## Run

```bash
uv run python 3-baseline-eval/run.py                                    # both suites
uv run python 3-baseline-eval/run.py --suite byod                       # BYOD only
uv run python 3-baseline-eval/run.py --target-model gpt-4.1 \
                                     --judge-model gpt-4.1               # override
```

## Output

```
3-baseline-eval/results/
  baseline_byod.json     # GPT-4.1 + BYOD numbers
  baseline_chat.json     # GPT-4.1 chat-only numbers
```

## Note on judge model

GPT-5.x rejects `temperature` from the azure-ai-evaluation SDK. Use **GPT-4.1
as the judge** until the SDK is updated.
