# Retrieval Architecture

**Release:** `v0.1.0 — Evaluated RAG Foundation`  
**Status:** Current implementation

This document describes retrieval actually included in v0.1.0.

## Pipeline

```text
User query
    ↓
Document-scoped retrieval request
    ↓
all-MiniLM-L6-v2
    ↓
ChromaDB similarity search
    ↓
Top-K chunks
    ↓
RAG generation / tool result
```

## Components

### Embeddings

`all-MiniLM-L6-v2` produces 384-dimensional embeddings and is suitable for the local-first constraint.

### Vector store

ChromaDB is the persistent local vector store.

Chunk metadata preserves provenance including:

- `document_id`
- `filename`
- `chunk_index`
- `chunk_size`
- `chunk_overlap`
- `embedding_model`
- `file_hash`
- `source_type`

### Chunking

```text
500 words / 50 word overlap
```

This is intentionally simple; structure-aware and token-aware strategies are future experiments.

## Document isolation

Document isolation is a correctness requirement.

A document-scoped query supplies a `document_id`, and retrieval is constrained to chunks belonging to that document.

```text
Query
  │
  ├── document_id = D
  ▼
ChromaDB
  ├── D / chunk 0
  ├── D / chunk 1
  └── ...
```

The official Phase 1A benchmark uses one target document and passes its identifier to the retriever.

## Frozen evaluation

Corpus:

```text
Attention Is All You Need
document_id = 99f9c01da0ebd7bf
```

Results:

| Metric | Result |
|---|---:|
| Hit@1 | 66.7% |
| Hit@3 | 86.7% |
| Hit@5 | 93.3% |
| MRR | 0.783 |
| Average latency | 15.5 ms |

The benchmark uses manually curated lexical evidence phrases and is intentionally a lightweight retrieval baseline.

## Not implemented in v0.1.0

- BM25
- Reciprocal Rank Fusion
- cross-encoder reranking
- adaptive retrieval
- retrieval-quality judging
- query rewriting/retry
- semantic relevance evaluation

These are future experiments and must be compared against the frozen dense baseline.

## Planned progression

```text
v0.1.0 Dense Chroma
       ↓
v0.2.0 Dense + BM25
       ↓
RRF candidate fusion
       ↓
v0.3.0 Cross-encoder reranking
       ↓
v0.4.0 Adaptive retrieval
       ↓
v0.5.0 Evidence self-correction
```

## Design principles

1. Isolation before sophistication.
2. Dense retrieval is the control.
3. Benchmark before adoption.
4. Preserve provenance.
5. Remain local-first.