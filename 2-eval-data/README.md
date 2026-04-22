# 2 — Eval data generation

Generates synthetic eval queries from documents in your Azure AI Search index,
then **validates** each query by running it back against the index and keeping
only those whose source document appears in the top-K results
([ARES](https://arxiv.org/abs/2311.09476) approach).

Pipeline: sample docs → LLM-generate queries → retrieval validation → JSONL.

## Run

```bash
uv run python 2-eval-data/generate.py                          # defaults
uv run python 2-eval-data/generate.py --retrieval-top-k 10     # stricter ARES
uv run python 2-eval-data/generate.py --gen-model gpt-4.1      # override
```

## Output

```
2-eval-data/data/
  byod_test_data.jsonl      # used by 3-baseline-eval and 5-experiments
  chat_test_data.jsonl
```

JSONL records have `query`, `ground_truth`, `context`, `doc_id` columns.
