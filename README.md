# DocuMind AI — Agentic RAG Document Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-1C3C3C?style=flat)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6B35?style=flat)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.1-F55036?style=flat)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **agentic RAG system** that lets you upload any PDF document and ask questions in natural language. A LLM-powered router dynamically selects the right tool — document retrieval, summarization, or calculation — rather than following a hardcoded pipeline. Responses are grounded in your documents, eliminating hallucinations.

> **Live demo:** [Hugging Face Spaces link] &nbsp;|&nbsp; **Blog post:** [Medium / Hashnode link]

---

## Why This Project

Most RAG tutorials stop at "embed → retrieve → answer." This project adds an **agentic decision layer**: the LLM examines the user's intent and routes to the appropriate tool on its own. This mirrors how production AI systems work at companies like Glean, Notion AI, and Perplexity — where a single query might require retrieval, computation, and synthesis in combination.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION (one-time)                    │
│                                                             │
│  PDF / TXT  →  Chunk (500 tokens)  →  Embed  →  ChromaDB  │
│               sentence-transformers                         │
└─────────────────────────────────────────────────────────────┘
                            │
                    persisted to disk
                            │
┌─────────────────────────────────────────────────────────────┐
│                     RUNTIME (per query)                     │
│                                                             │
│   User query                                                │
│       │                                                     │
│       ▼                                                     │
│   Agentic Router  (Llama 3.1 8B via Groq function calling) │
│       │                                                     │
│   ┌───┼───────────────┐                                     │
│   ▼   ▼               ▼                                     │
│  RAG  Calculator   Summarize                                │
│  Tool    Tool        Tool                                   │
│   │                   │                                     │
│   └──── ChromaDB ─────┘                                     │
│         (similarity search, top-k=3)                        │
│                                                             │
│       ▼                                                     │
│   Final answer  +  source citations                         │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Tool-use over chains** — the router uses Groq's function-calling API so the LLM selects tools based on intent, not keyword matching. A question like *"What is 15% of the budget mentioned in the report?"* triggers both RAG retrieval and the calculator in sequence.
- **Local vector store** — ChromaDB persists to disk. No database server, no API cost, no data leaving your machine.
- **Pluggable LLM** — Groq (cloud, free tier) by default; swap to Ollama + Phi-3 mini for fully offline use on modest hardware (tested on GTX 1650 Ti / 4GB VRAM).
- **Conversation memory** — full chat history is passed on each turn, enabling multi-turn follow-up questions without re-ingesting documents.

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| LLM | Groq API (Llama 3.1 8B) | Free tier: 30 req/min |
| Local LLM fallback | Ollama + Phi-3 mini | Runs on 4GB VRAM |
| Embeddings | `all-MiniLM-L6-v2` | CPU-friendly, 90MB |
| Vector store | ChromaDB (persistent) | Local, zero cost |
| PDF parsing | PyMuPDF (fitz) | Fast, handles scanned PDFs |
| Agentic loop | LangGraph / bare Python | Transparent tool routing |
| UI | Gradio | Deployable to HF Spaces |
| Experiment tracking | MLflow (optional) | Log retrieval quality metrics |

---

## Features

- **Multi-tool agentic routing** — LLM decides between RAG search, math calculation, and summarization
- **PDF ingestion pipeline** — upload any PDF via UI; chunked and embedded in under 10 seconds
- **Source citations** — every answer references the document chunks it was drawn from
- **Conversation memory** — multi-turn Q&A without re-ingesting context
- **Fully local mode** — swap Groq for Ollama; no data leaves your machine
- **Drift-aware chunking** — chunk metadata stored for easy re-ingestion when documents update

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/jmhasan1/documind-ai
cd documind-ai
pip install -r requirements.txt

# 2. Set your Groq API key (free at console.groq.com)
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key_here

# 3. (Optional) Use local LLM instead
# Install Ollama: https://ollama.com
# ollama pull phi3:mini
# Set USE_LOCAL_LLM=true in .env

# 4. Run
python app.py
# Opens at http://localhost:7860
```

---

## Project Structure

```
documind-ai/
├── app.py                  ← Gradio UI entry point
├── agent.py                ← Agentic router + tool definitions
├── rag_core.py             ← Retrieval + generation
├── ingest.py               ← PDF → chunk → embed → ChromaDB
├── tools/
│   ├── rag_tool.py         ← Similarity search + LLM answer
│   ├── calculator_tool.py  ← Safe math expression evaluator
│   └── summarize_tool.py   ← Document-grounded summarization
├── notebooks/
│   └── exploration.ipynb   ← Chunking experiments, retrieval eval
├── data/
│   └── sample_docs/        ← Sample PDFs for testing
├── requirements.txt
├── .env.example
└── chroma_db/              ← gitignored; created on first ingest
```

---

## How the Agentic Loop Works

A standard RAG system always retrieves and always answers the same way. This system adds a **decision layer**:

```
User: "What is 20% of the revenue figure mentioned in the Q3 report?"

Router sees: math + document lookup required
  → calls rag_tool("Q3 revenue figure")     → retrieves "$4.2M"
  → calls calculator_tool("4200000 * 0.20") → returns "840000"
  → synthesizes: "20% of the Q3 revenue of $4.2M is $840,000"
```

The router uses **Groq's native function-calling API** — no prompt-engineering hacks, no regex parsing. The LLM returns a structured JSON tool call, the tool executes, and the result is fed back for final synthesis.

---

## Retrieval Quality

| Metric | Value | Notes |
|---|---|---|
| Chunk size | 500 tokens | ~350 words; balanced recall vs precision |
| Overlap | 50 tokens | Prevents context loss at boundaries |
| Top-k retrieval | 3 | Sufficient context without token bloat |
| Embedding model | MiniLM-L6-v2 | 384-dim; fast on CPU |
| Similarity metric | Cosine | Default ChromaDB |

Retrieval evaluation notebooks are in `notebooks/exploration.ipynb` — includes precision@k and answer faithfulness checks using a small hand-labeled test set.

---

## Extending This Project

This project is designed as a foundation. Planned extensions:

- **Network Security RAG** — layer this retrieval system over the [Network Security ML Pipeline](https://github.com/jmhasan1/Network-Security-ML-Pipeline) to explain *why* a connection was flagged as malicious, drawing from a vector store of known phishing patterns
- **Agentic Recommender** — reuse the tool-routing skeleton with `get_user_history` and `fetch_item_metadata` tools for a personalized recommendation agent
- **Reranking** — add a cross-encoder reranker (e.g. `ms-marco-MiniLM`) between retrieval and generation for higher precision
- **Evaluation pipeline** — RAGAS framework integration for automated RAG quality scoring

---

## Running Locally Without GPU

Tested on an i5 10th Gen, 16GB RAM, GTX 1650 Ti (4GB VRAM):

- Embeddings run on CPU; ~150–200ms per batch — no GPU required
- `phi3:mini` via Ollama uses ~3.8GB VRAM; fits the 1650 Ti
- ChromaDB is disk-based; no GPU involvement
- Groq API (default) offloads inference entirely — recommended for development

---

## License

MIT — see [LICENSE](LICENSE)

---

## Connect

**Jahid Md Hasan** — [LinkedIn](https://www.linkedin.com/in/jahid-hasan-19jm/) · [GitHub](https://github.com/jmhasan1) · [Portfolio](https://jmhasan1.github.io/Jahid_Portfolio/) · jmhasan17@gmail.com
