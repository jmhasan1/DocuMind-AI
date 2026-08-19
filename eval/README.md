# DocuMind Evaluation — Phase 1A

This directory establishes the **baseline before retrieval upgrades**.

## Current baseline

The evaluator measures the existing dense retriever:

```text
Question
  ↓
all-MiniLM-L6-v2
  ↓
ChromaDB
  ↓
Top-K
```

Metrics:

- Hit@1
- Hit@3
- Hit@5
- MRR
- average retrieval latency

## Run

From the project root, after ingesting the bundled evaluation PDF:

```bash
uv run python -m eval.run_retrieval_baseline
```

## Important limitation

The first baseline uses manually curated **evidence phrases** rather than full human relevance judgments. This keeps the first experiment lightweight and reproducible.

It is intentionally not presented as a final RAG evaluation methodology.

Later phases will add:

- chunk-level relevance judgments
- semantic retrieval evaluation
- faithfulness
- answer relevance
- citation correctness
- adversarial/unanswerable questions
- regression testing

The key rule is: **every retrieval upgrade must be compared against this baseline.**

### Phase 1A Baseline
** This is the official document-scoped Phase 1A baseline. **

Evaluation corpus:
Attention Is All You Need

Document ID:
99f9c01da0ebd7bf

Queries:
15

Retriever:
Dense Chroma + all-MiniLM-L6-v2

Chunking:
500 / 50

Results:

Hit@1: 66.7% (10/15)
Hit@3: 86.7% (13/15)
Hit@5: 93.3% (14/15)
MRR: 0.783
Average retrieval latency: 30.4 ms
