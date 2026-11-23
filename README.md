# 🧠 DocuMind AI — Agentic RAG Document Assistant

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=flat)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6B35?style=flat)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=flat)](https://groq.com/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991?style=flat)](https://openai.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Local-first, agentic RAG for document intelligence — with a reproducible retrieval evaluation baseline.**

DocuMind AI is a local-first Retrieval-Augmented Generation (RAG) document assistant. It ingests PDF documents, stores local embeddings in ChromaDB, retrieves relevant chunks, and uses an LLM to produce document-grounded responses. The application also exposes calculator and document-summarization tools through an agent loop and supports OpenAI/Groq provider switching.

The project is being developed as a **Regulatory & Compliance Document Intelligence** system. Future retrieval, grounding, evaluation, security, governance, and document-lifecycle work should solve concrete document-intelligence problems rather than add architecture for its own sake.

---

## Why This Project?

DocuMind is being developed as an experimental document-intelligence system where retrieval changes are treated as measurable experiments. Each retrieval upgrade is evaluated against a frozen baseline rather than being accepted solely because it is architecturally more sophisticated.<br>
<br>A conventional RAG system usually follows:

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

This enables questions that require more than one capability. For example:

> **What was the Q3 revenue mentioned in the report, and what is 20% of it?**

The agent can retrieve the revenue, calculate the percentage, and synthesize the result.

---

## v0.1.0 — Evaluated RAG Foundation

This release freezes the first evaluated foundation.

### Included

- PDF ingestion with PyMuPDF
- 500-word chunks with 50-word overlap
- `all-MiniLM-L6-v2` embeddings
- Persistent ChromaDB vector storage
- Document/chunk provenance metadata
- Document-scoped dense retrieval
- Agentic tool routing
- Calculator and document-summarization tools
- Multi-turn conversation support
- OpenAI/Groq provider abstraction through LangChain
- Reproducible 15-query retrieval benchmark
- Frozen dense-retrieval metrics

### Deliberately not included

These remain future experiments:

- BM25 sparse retrieval
- Hybrid retrieval / RRF
- Cross-encoder reranking
- Adaptive retrieval
- Retrieval self-correction / grounding judges
- Multi-agent retrieval
- GraphRAG
- Production observability/security/governance layers

**Rule:** every retrieval upgrade is measured against the frozen dense baseline before claiming an improvement.

## Architecture

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
                 │                      Dense retrieval
                 ▼                           │
             ChromaDB                       ▼
                 │                       Top-K chunks
                 └──────────────────────────┘
                                    │
                                    ▼
                                Final LLM
                                    │
                                    ▼
                                  Answer
```

At v0.1.0 retrieval is **dense-only**. The official evaluator passes each query's `document_id` to retrieval, preventing other documents from entering the benchmark search scope.

## Evaluation baseline

The official benchmark contains **15 curated questions** over `Attention Is All You Need`.

| Metric | v0.1.0 |
|---|---:|
| Hit@1 | **66.7% (10/15)** |
| Hit@3 | **86.7% (13/15)** |
| Hit@5 | **93.3% (14/15)** |
| MRR | **0.783** |
| Avg. retrieval latency | **24.4 ms** |

The benchmark uses manually curated lexical evidence phrases rather than full human relevance judgments. It is a retrieval baseline, not a final end-to-end RAG quality evaluation.

Baseline artifact:

```text
eval/results/v0.1_dense_baseline.json
```

Historical tag:

```text
v0.1-dense-baseline
```

Release:

```text
v0.1.0 — Evaluated RAG Foundation
```

## Evaluation

```powershell
uv run python -m eval.run_retrieval_baseline
```

The evaluator:

1. loads the 15-query dataset;
2. reads each query's target `document_id`;
3. retrieves only from that document;
4. checks results against curated evidence phrases;
5. computes Hit@1, Hit@3, Hit@5, MRR, and average latency.

## Project structure

```text
DocuMind_AI/
├── app.py
├── agent.py
├── llm_factory.py
├── rag_core.py
├── ingest.py
├── data/
├── chroma_db/                 # local; not committed
├── eval/
│   ├── dataset.jsonl
│   ├── run_retrieval_baseline.py
│   ├── README.md
│   └── results/
│       └── v0.1_dense_baseline.json
├── docs/
│   ├── ROADMAP.md
│   ├── architecture/
│   │   └── retrieval.md
│   └── decisions/
│       └── 0001-domain-framing.md
├── scripts/
├── notebooks/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

Requirements:

- Python 3.13
- `uv`
- Groq and/or OpenAI API credentials
- 8 GB+ RAM recommended
- GPU is not required for embeddings

```powershell
git clone https://github.com/jmhasan1/DocuMind-AI.git
cd DocuMind-AI
uv sync
```

Configure `.env` locally:

```env
LLM_PROVIDER=groq

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.6-27b
```

Never commit `.env` or API credentials.

## Release roadmap

```text
v0.1.0  Evaluated RAG Foundation
   ↓
v0.2.0  Hybrid Retrieval
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

The exact scope of future releases is governed by experimental results.

## Design principles

1. **Local-first**
2. **Document isolation**
3. **Measure before claiming**
4. **Reproducible evaluation**
5. **Evidence over demo quality**
6. **Domain-driven architecture**

## Testing

DocuMind AI uses **pytest** for automated testing.

Run the full test suite:

```bash
uv run pytest -v
```

The test suite covers environment validation, retrieval, RAG answer generation, and agent behavior.

**Current status:** 14 tests passed.

Document-level retrieval isolation is also validated through manual UI testing, ensuring that queries against one active document do not retrieve content from another document.


## License

MIT.

