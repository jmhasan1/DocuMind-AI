# DocuMind Evaluation — Phase 1A

This directory establishes the **frozen retrieval baseline before retrieval upgrades**.

## Baseline pipeline

```text
Question
   ↓
all-MiniLM-L6-v2
   ↓
ChromaDB
   ↓
Document-scoped Top-K
   ↓
Evidence-phrase evaluation
```

Metrics:

- Hit@1
- Hit@3
- Hit@5
- MRR
- average retrieval latency (Note: latency is a runtime observation, not a stable benchmark claim)

## Run

```powershell
uv run python -m eval.run_retrieval_baseline
```

Each query contains a target `document_id`, which is passed into retrieval.

## Official v0.1.0 baseline

Evaluation corpus:

```text
Attention Is All You Need
```

Document ID:

```text
99f9c01da0ebd7bf
```

Queries: **15**

Retriever:

```text
ChromaDB dense retrieval + all-MiniLM-L6-v2
```

Chunking:

```text
500 words / 50 word overlap
```

### Results

| Metric | v0.1.0 |
|---|---:|
| Hit@1 | **66.7% (10/15)** |
| Hit@3 | **86.7% (13/15)** |
| Hit@5 | **93.3% (14/15)** |
| MRR | **0.783** |
| Average retrieval latency | **24.4 ms** |

Result artifact:

```text
eval/results/v0.1_dense_baseline.json
```

## Metric definitions

**Hit@K:** whether at least one relevant result appears within the top K.

**MRR:** mean reciprocal rank of the first relevant result. For rank `r`, reciprocal rank is `1/r`.

**Latency:** average elapsed retrieval/evaluation time per query.

Higher Hit@K/MRR is better; lower latency is generally better when quality is maintained.

## Methodology limitation

The dataset uses manually curated **lexical evidence phrases**. A retrieved chunk is considered relevant when one of the expected phrases occurs in it.

This is a lightweight, deterministic baseline—not a final semantic RAG evaluation.

It does not yet establish:

- graded relevance
- semantic equivalence
- answer faithfulness
- answer relevance
- citation correctness
- adversarial/unanswerable behavior

Those are future evaluation stages.

## Baseline policy

Every retrieval upgrade should report the same core metrics against this frozen benchmark before changing evaluation semantics.

```text
Hit@1
Hit@3
Hit@5
MRR
Average latency
```

## Release naming

`v0.1_dense_baseline` is the frozen dense-retrieval experiment.

The project release is:

```text
v0.1.0 — Evaluated RAG Foundation
```

The historical Git tag is:

```text
v0.1-dense-baseline
```