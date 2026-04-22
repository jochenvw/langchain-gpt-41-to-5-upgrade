# Prompt: eval_data_multi_doc

**Used by**: `2-eval-data/generate.py` — multi-document (cross-source) query generation.

**Purpose**: Synthesize queries that REQUIRE evidence from two source documents — used to stress retrieval recall in evals.

**Variables**: none.

---

You are an evaluation dataset generator. You are given TWO document excerpts.
Generate exactly 2 realistic questions that REQUIRE information from BOTH documents to answer fully.
The question should need facts from Document A AND Document B — a single document alone should not
be sufficient for a complete answer.

Also provide a concise ground-truth answer for each question, citing relevant facts from both documents.

Return a JSON array of objects with "query" and "ground_truth" keys. No markdown fencing.

Example output:
[{"query": "How do the PPE requirements differ between confined spaces and hot work areas?", "ground_truth": "Confined spaces require respiratory protection and atmospheric monitoring, while hot work areas require fire-resistant clothing and a dedicated fire watch."}]
