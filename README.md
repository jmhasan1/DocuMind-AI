# 🧠 DocuMind AI — Agentic RAG Document Assistant

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=flat)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6B35?style=flat)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=flat)](https://groq.com/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991?style=flat)](https://openai.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Local-first, agentic RAG for document intelligence — built around reproducible retrieval evaluation.**

DocuMind AI is a local-first Retrieval-Augmented Generation (RAG) document assistant. It ingests PDF documents, stores embeddings in ChromaDB, retrieves relevant chunks, and uses an LLM to produce document-grounded responses. The application also exposes calculator and document-summarization tools through an agent loop and supports OpenAI/Groq provider switching.

The project is being developed as a **Regulatory & Compliance Document Intelligence** system. Retrieval, grounding, evaluation, security, governance, and document-lifecycle improvements are guided by measurable document-intelligence requirements rather than architectural complexity for its own sake.

<text>**Current development:** v0.2.0 — Hybrid Retrieval (Sprint 1 in progress).</text>

---

## Why this project?

DocuMind treats retrieval changes as measurable experiments. Each retrieval upgrade is evaluated against a frozen baseline rather than accepted simply because it introduces a more sophisticated architecture.

A conventional RAG system follows:

```text
Question → Retrieve → Generate
```

DocuMind adds an agentic decision layer:

```text
Question
   │
   ▼
 Agent
   │
   ├──► RAG Search
   ├──► Calculator
   └──► Summarize Document
           │
           ▼
      Tool Results
           │
           ▼
     Final LLM Answer
```

For example:

> **What was the Q3 revenue mentioned in the report, and what is 20% of it?**

The agent can retrieve the revenue, calculate the percentage, and synthesize the result.

---

## Project status

### v0.1.0 — Evaluated RAG Foundation

<badge color="success">Frozen baseline</badge>

The first evaluated foundation includes:

- PDF ingestion with PyMuPDF.
- 500-word chunks with 50-word overlap.
- `all-MiniLM-L6-v2` embeddings.
- Persistent ChromaDB vector storage.
- Document/chunk provenance metadata.
- Document-scoped dense retrieval.
- Agentic tool routing.
- Calculator and document-summarization tools.
- Multi-turn conversation support.
- OpenAI/Groq provider abstraction through LangChain.
- Reproducible 15-query retrieval benchmark.

### v0.2.0 — Hybrid Retrieval

<badge color="warning">In progress</badge>

The current Sprint 1 work adds sparse retrieval alongside the frozen dense baseline.

**Implemented and tested:**

- Dense retrieval extracted into `retrieval/dense.py`.
- Backward-compatible `rag_core.retrieve_chunks()` wrapper.
- BM25 sparse retrieval using `rank-bm25` and `BM25Okapi`.
- ChromaDB as the canonical persistent chunk source.
- Lazy, in-memory BM25 indexes cached per document scope.
- Cache invalidation after successful ingestion.
- Sparse retrieval and ingestion unit tests.
- Document-scope handling in the sparse retrieval component.

**Still pending:**

- Reciprocal Rank Fusion (RRF).
- Unified retrieval pipeline.
- `vector`, `bm25`, and `hybrid` retrieval modes.
- Integrated document-isolation tests across all modes.
- Comparative dense/BM25/hybrid benchmark.
- Reproducible v0.2 experiment artifact.

BM25 is currently an independently tested retrieval component. The complete hybrid pipeline is not yet integrated.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for release scope and exit criteria.

---

## Architecture

### Current retrieval components

```text
                         ┌──────────────────────┐
                         │      Gradio UI       │
                         │  Chat + PDF Upload   │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
          Document ingestion                       User query
                 │                                     │
                 ▼                                     ▼
              PyMuPDF                                Agent
                 │                                     │
                 ▼                          ┌──────────┼──────────┐
          Chunking + metadata               │          │          │
                 │                        RAG Search  Calc   Summarize
                 ▼                           │
        MiniLM embeddings                   ▼
                 │                   Dense retrieval
                 ▼                           │
             ChromaDB                       ▼
                 ▲                     Retrieved chunks
                 │
          BM25 corpus source
                 │
                 ▼
          Sparse retrieval
          (implemented component)

```

The dense and BM25 retrievers are currently separate components. RRF fusion and the unified retrieval pipeline are planned work, not part of the current integrated application flow.

### Intended v0.2 hybrid retrieval

```text
Question
   │
   ├── Dense retriever ── top-N ──┐
   │                              │
   └── BM25 retriever ─── top-N ──┤
                                  ▼
                             RRF fusion
                                  │
                                  ▼
                             Final top-K
```

This is the target architecture for Sprint 1. It should be considered implemented only after the pipeline and its tests are complete.

---

## v0.1.0 retrieval evaluation baseline

The frozen benchmark contains **15 curated questions** over *Attention Is All You Need*.

| Metric | v0.1.0 |
|---|---:|
| Hit@1 | **66.7% (10/15)** |
| Hit@3 | **86.7% (13/15)** |
| Hit@5 | **93.3% (14/15)** |
| MRR | **0.783** |
| Average retrieval latency | **24.4 ms** |

The benchmark uses manually curated lexical evidence phrases rather than full human relevance judgments. It is a retrieval baseline, not a final end-to-end RAG quality evaluation. Latency is run-specific and should be interpreted with its original environment and measurement conditions.

**Baseline artifact:**

```text
eval/results/v0.1_dense_baseline.json
```

**Historical tag:**

```text
v0.1-dense-baseline
```

**Release:**

```text
v0.1.0 — Evaluated RAG Foundation
```

The v0.1 baseline is frozen. Future experiments must not overwrite its artifact or silently change its dataset, corpus, chunking, or relevance semantics.

## Evaluation

Run the frozen dense retrieval evaluator:

```powershell
uv run python -m eval.run_retrieval_baseline
```

The evaluator:

1. Loads the 15-query dataset.
2. Reads each query's target `document_id`.
3. Retrieves only from that document.
4. Checks results against curated evidence phrases.
5. Computes Hit@1, Hit@3, Hit@5, MRR, and average latency.

The v0.2 experiment will compare dense-only, BM25-only, and hybrid RRF retrieval using the same evaluation basis.

---

## Project structure

```text
DocuMind_AI/
├── app.py
├── agent.py
├── llm_factory.py
├── rag_core.py
├── ingest.py
├── retrieval/
│   ├── dense.py
│   └── sparse.py
├── data/
├── chroma_db/                 # local; not committed
├── eval/
│   ├── dataset.jsonl
│   ├── run_retrieval_baseline.py
│   ├── README.md
│   └── results/
│       └── v0.1_dense_baseline.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DECISIONS.md
│   ├── DEVELOPMENT_LOG.md
│   ├── PROJECT_CONTEXT.md
│   ├── ROADMAP.md
│   ├── testing.md
│   ├── architecture/
│   │   └── retrieval.md
│   └── decisions/
│       ├── 0001-domain-framing.md
│       ├── 0002-hybrid-retrieval.md
│       ├── 0003-document-isolation.md
│       └── 0004-bm25-index-lifecycle.md
├── tests/
│   ├── test_environment.py
│   └── unit/
│       ├── test_agent.py
│       ├── test_ingest.py
│       ├── test_rag_answer.py
│       ├── test_retrieval.py
│       └── test_sparse_retrieval.py
├── scripts/
├── notebooks/
├── AGENTS.md
├── pyproject.toml
├── uv.lock
└── README.md
```

The tree highlights the main project components. Consult the repository for the complete current file layout.

---

## Installation

### Requirements

- Python 3.13.
- [`uv`](https://docs.astral.sh/uv/).
- Groq and/or OpenAI API credentials.
- 8 GB+ RAM recommended.
- GPU is not required for the baseline embedding workflow.

### Clone and install

```powershell
git clone https://github.com/jmhasan1/DocuMind-AI.git
cd DocuMind-AI
uv sync
```

### Configure environment variables

Create a local `.env` file:

```env
LLM_PROVIDER=groq

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.6-27b
```

Use the provider and model names supported by your current project configuration.

**Never commit `.env` files or API credentials.**

---

## Testing

DocuMind AI uses `pytest` for automated testing.

Run the complete test suite:

```powershell
uv run pytest -v
```

Run focused retrieval tests:

```powershell
uv run pytest -v tests/unit/test_retrieval.py
uv run pytest -v tests/unit/test_sparse_retrieval.py
uv run pytest -v tests/unit/test_ingest.py
```

The latest reported Sprint 1 result is:

```text
31 passed
0 failed
1 non-blocking AnyIO deprecation warning
```

This is the latest reported test result, not a guarantee that all pending hybrid integration work is complete. See [`docs/testing.md`](docs/testing.md) for test coverage, validation status, and remaining Sprint 1 requirements.

---

## Release roadmap

```text
v0.1.0  Evaluated RAG Foundation        [FROZEN]
   ↓
v0.2.0  Hybrid Retrieval                [IN PROGRESS]
   ↓
v0.3.0  Reranking & Retrieval Evaluation
   ↓
v0.4.0  Adaptive RAG
   ↓
v0.5.0  Self-Correction / Grounding
   ↓
v0.6.0  Agentic RAG V2
   ↓
v0.7.0  Multi-Agent RAG
   ↓
v0.8.0  Evaluation & Observability Platform
   ↓
v0.9.0  Production Hardening
   ↓
v1.0.0  Production-Oriented Document Intelligence System
```

The exact scope of future releases is governed by experimental results and documented release criteria.

---

## Design principles

1. **Local-first:** prioritize practical local execution and resource awareness.
2. **Document isolation:** prevent retrieval across unauthorized document scopes.
3. **Measure before claiming:** use controlled experiments to evaluate improvements.
4. **Reproducible evaluation:** preserve datasets, configurations, and historical artifacts.
5. **Evidence over demo quality:** prioritize grounded answers and traceable evidence.
6. **Domain-driven architecture:** solve concrete regulatory and compliance document problems.
7. **Incremental complexity:** introduce advanced retrieval and agentic components only when justified.

## Documentation

- [Project roadmap](docs/ROADMAP.md)
- [Architecture overview](docs/ARCHITECTURE.md)
- [Retrieval architecture](docs/architecture/retrieval.md)
- [Architecture decisions](docs/DECISIONS.md)
- [Testing guide](docs/testing.md)
- [Development log](docs/DEVELOPMENT_LOG.md)
- [Project context](docs/PROJECT_CONTEXT.md)
- [Codex instructions](AGENTS.md)

## License

MIT.

