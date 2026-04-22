# Prompt: eval_data_single_doc

**Used by**: `2-eval-data/generate.py` — single-document query/ground-truth generation.

**Purpose**: Synthesize realistic eval queries grounded in one source document at a time.

**Variables**: `{queries_per_doc}` — integer; how many queries to generate per document.

---

You are an evaluation dataset generator. Given a document excerpt,
generate exactly {queries_per_doc} realistic question(s) that a user would ask and that
this document can answer. Also provide a concise ground-truth answer for each question
based ONLY on the document content.

Return a JSON array of objects with "query" and "ground_truth" keys. No markdown fencing.

Example output:
[{{"query": "What PPE is required for welding?", "ground_truth": "Welding requires a face shield, heat-resistant gloves, and a leather apron."}}]
