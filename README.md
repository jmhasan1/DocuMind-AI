# 🧠 DocuMind AI — Agentic RAG Document Assistant

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=flat)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6B35?style=flat)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=flat)](https://groq.com/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991?style=flat)](https://openai.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DocuMind AI** is a local-first, agentic Retrieval-Augmented Generation (RAG) document assistant. Upload a PDF, ingest it into a persistent ChromaDB vector store, and ask questions in natural language.

The system exposes multiple tools to the LLM and lets the model decide which capability is required:

- 🔎 **RAG Search** — retrieve relevant document chunks
- 🧮 **Calculator** — evaluate mathematical expressions
- 📝 **Document Summarization** — generate document-grounded summaries
- 💬 **Conversation Memory** — support multi-turn follow-up questions
- 🔄 **Provider Switching** — switch between **Groq** and **OpenAI** from the Gradio UI

The project uses **LangChain's provider-neutral chat model interface**. Both `ChatGroq` and `ChatOpenAI` use the same `.invoke()` / `.bind_tools()` architecture. The application does **not** directly call `client.chat.completions.create()`.

---

## Why This Project?

A conventional RAG system usually follows:

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

## Architecture

```text
                         ┌──────────────────────┐
                         │      Gradio UI       │
                         │  Chat + PDF Upload   │
                         └──────────┬───────────┘
                                    │
                         Provider Selection
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                  ┌─────────────┐       ┌─────────────┐
                  │ ChatOpenAI  │       │  ChatGroq   │
                  │   OpenAI    │       │    Groq     │
                  └──────┬──────┘       └──────┬──────┘
                         │                     │
                         └──────────┬──────────┘
                                    │
                              LangChain API
                                    │
                         ┌──────────▼──────────┐
                         │    Agent Loop       │
                         │  bind_tools()       │
                         │  invoke()           │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              ┌──────────┐   ┌────────────┐   ┌──────────────┐
              │ RAG Tool │   │ Calculator │   │ Summarizer   │
              └────┬─────┘   └────────────┘   └──────┬───────┘
                   │                                  │
                   ▼                                  ▼
             ┌───────────┐                     ┌─────────────┐
             │ ChromaDB  │                     │ ChromaDB +  │
             │ Retrieval │                     │ LLM Context │
             └─────┬─────┘                     └──────┬──────┘
                   │                                  │
                   └────────────────┬─────────────────┘
                                    ▼
                             Tool Results
                                    │
                                    ▼
                              Final LLM Answer
                                    │
                                    ▼
                               Gradio Chat
```

### Ingestion pipeline

```text
PDF → PyMuPDF → Text → Chunking → all-MiniLM-L6-v2 → ChromaDB
```

### Runtime pipeline

```text
User Query
    ↓
Provider Factory
    ↓
ChatOpenAI / ChatGroq
    ↓
Agent + bind_tools()
    ↓
RAG / Calculator / Summarizer
    ↓
Tool Results
    ↓
LLM.invoke()
    ↓
Final Answer
```

---

## OpenAI ↔ Groq Provider Switching

The provider layer is centralized in `llm_factory.py`.

```python
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


def get_llm(provider: str):
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

    if provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

    raise ValueError(f"Unsupported provider: {provider}")
```

The rest of the agent does not need provider-specific API code:

```python
llm = get_llm(provider)
response = llm.invoke(messages)
```

For tools:

```python
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke(messages)
```

This avoids the previous error:

```text
AttributeError: 'ChatOpenAI' object has no attribute 'chat'
```

The incorrect pattern was:

```python
llm.chat.completions.create(...)
```

The correct LangChain pattern is:

```python
llm.invoke(...)
```

---

## Agentic Tool Loop

The agent can select tools based on the user's intent.

```text
User Question
     │
     ▼
Selected LLM
(ChatOpenAI / ChatGroq)
     │
     ▼
.bind_tools()
     │
     ▼
Model decides whether tools are needed
     │
 ┌───┼───────────────┐
 ▼   ▼               ▼
RAG Calculator   Summarizer
 │   │               │
 └───┴───────────────┘
          │
          ▼
      Tool Results
          │
          ▼
     Final .invoke()
          │
          ▼
      Final Answer
```

A multi-step request can therefore become:

```text
User
 ↓
RAG Search → "$4.2M"
 ↓
Calculator → "$840,000"
 ↓
Final synthesis
```

---

## Conversation Memory

Gradio provides previous turns to the chat function. The application sanitizes the history and passes it to the agent as conversation context.

```text
Turn 1
User → HumanMessage
Assistant → AIMessage

Turn 2
User → HumanMessage
        ↓
  Full conversation history
        ↓
 ChatOpenAI / ChatGroq
```

This supports follow-up questions such as:

```text
User: What is the main objective?
Assistant: ...

User: Who is responsible for it?
Assistant: ...
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Gradio | Chat interface and PDF upload |
| Agent framework | LangChain | Model/tool abstraction |
| OpenAI integration | `langchain-openai` | `ChatOpenAI` |
| Groq integration | `langchain-groq` | `ChatGroq` |
| Embeddings | Sentence Transformers | Local embeddings |
| Embedding model | `all-MiniLM-L6-v2` | CPU-friendly semantic embeddings |
| Vector database | ChromaDB | Persistent local vector store |
| PDF parser | PyMuPDF | PDF text extraction |
| Calculator | NumExpr | Mathematical evaluation |
| Configuration | python-dotenv | Environment variables |
| Package manager | uv | Reproducible environment |

---

## Retrieval Configuration

| Parameter | Value |
|---|---:|
| Chunk size | 500 words |
| Chunk overlap | 50 words |
| Top-k retrieval | 3 |
| Embedding model | `all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Vector store | ChromaDB |
| Persistence | Local disk |

The current chunker is intentionally simple and lightweight. Future versions can add token-aware, structure-aware, hybrid retrieval, and reranking.

---

## Project Structure

```text
DocuMind_AI/
│
├── app.py                  # Gradio UI and document ingestion trigger
├── agent.py                # Agent loop and tool routing
├── llm_factory.py          # OpenAI/Groq provider abstraction
├── rag_core.py             # Retrieval and grounded generation
├── ingest.py               # PDF parsing, chunking, embeddings, ChromaDB
│
├── data/
│   └── your_document.pdf   # Example/local document
│
├── chroma_db/              # Persistent vector database; gitignored
│
├── .env                    # Local API credentials; never commit
├── .env.example            # Environment variable template
├── .gitignore
├── pyproject.toml          # Project metadata/dependencies
├── uv.lock                 # Locked dependency resolution
├── requirements.txt        # Optional compatibility file
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.13**
- `uv`
- Groq and/or OpenAI API key
- 8 GB+ RAM recommended
- GPU is not required for embeddings

Install `uv` if needed:

```powershell
pip install uv
```

Clone the project:

```powershell
git clone https://github.com/jmhasan1/DocuMind-AI.git
cd DocuMind-AI
```

Create/synchronize the environment:

```powershell
uv sync
```

---

## Environment Configuration

Create `.env` from `.env.example`:

```env
LLM_PROVIDER=groq

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

You only need the credentials for the provider you use, but keeping both configured allows switching between providers in the UI.

### Security

Never commit `.env`.

Recommended `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
chroma_db/
```

If an API key is ever exposed, revoke/rotate it immediately.

---

## Run the Application

```powershell
uv run python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

---

## Verify the Providers Before Running Gradio

### Verify imports

```powershell
python -c "from langchain_openai import ChatOpenAI; from langchain_groq import ChatGroq; print('LangChain providers OK')"
```

Expected:

```text
LangChain providers OK
```

### Verify provider factory

```powershell
python -c "from llm_factory import get_llm; print(type(get_llm('groq')).__name__); print(type(get_llm('openai')).__name__)"
```

Expected:

```text
ChatGroq
ChatOpenAI
```

### Verify Groq invocation

```powershell
python -c "from llm_factory import get_llm; print(get_llm('groq').invoke('Say hello in one short sentence.').content)"
```

### Verify OpenAI invocation

```powershell
python -c "from llm_factory import get_llm; print(get_llm('openai').invoke('Say hello in one short sentence.').content)"
```

---

## Using DocuMind

### 1. Upload a PDF

Use **Upload Reference PDF** and select a PDF.

### 2. Ingest the document

Click **🚀 Ingest Document**.

```text
PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
Embedding
 ↓
ChromaDB
```

### 3. Select a provider

Choose **Groq** or **OpenAI** in the provider selector.

### 4. Ask questions

Examples:

```text
What is this document about?
What are the main objectives?
What does the document say about revenue?
Summarize the section about risk management.
What is 15% of the budget mentioned in the document?
```

---

## Example: RAG + Calculation

If the document contains:

```text
Q3 revenue was $4.2 million.
```

and the user asks:

```text
What is 20% of the Q3 revenue mentioned in the report?
```

the intended flow is:

```text
User question
     ↓
Agent
     ↓
RAG Search
     ↓
$4.2 million
     ↓
Calculator
     ↓
$840,000
     ↓
Final LLM synthesis
```

---

## Dependency Management with uv

This project uses `pyproject.toml` and `uv.lock` as the primary dependency sources.

Recommended workflow:

```powershell
uv sync
```

After dependency changes:

```powershell
uv lock
uv sync
```

Do **not** force incompatible versions of the standalone `groq` package when using `langchain-groq`; let the LangChain integration resolve a compatible Groq SDK version.

For this project, `requirements.txt` is retained mainly for compatibility/reference. The preferred development workflow is `uv sync`.

---

## Troubleshooting

### `ChatOpenAI object has no attribute 'chat'`

Cause: mixing the raw OpenAI SDK interface with LangChain.

Incorrect:

```python
llm.chat.completions.create(...)
```

Correct:

```python
response = llm.invoke(messages)
```

For tools:

```python
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke(messages)
```

### Dependency conflict involving `groq`

Check the dependency graph:

```powershell
uv pip list | Select-String "langchain|groq|openai"
```

If the lock file needs to be rebuilt:

```powershell
Remove-Item uv.lock
uv lock
uv sync
```

### Hugging Face Hub authentication warning

You may see an unauthenticated Hugging Face Hub warning while downloading the Sentence Transformers model. This is a rate-limit/authentication warning, not the cause of the `ChatOpenAI` runtime error.

---

## Hardware Considerations

The project is designed for modest local hardware. A representative development setup is:

```text
CPU: Intel Core i5 10th Gen
RAM: 16 GB
GPU: NVIDIA GTX 1650 Ti 4 GB
```

Local machine:

```text
PDF parsing
Chunking
Embeddings
ChromaDB
Agent orchestration
```

Cloud:

```text
LLM inference
├── Groq
└── OpenAI
```

This avoids requiring a large local GPU for LLM inference.

---

## Roadmap

### Phase 1 — Core System

- [x] PDF ingestion
- [x] Text chunking
- [x] Local embeddings
- [x] Persistent ChromaDB
- [x] RAG retrieval
- [x] Calculator tool
- [x] Summarization tool
- [x] Gradio chat UI
- [x] Conversation history
- [x] LangChain `ChatOpenAI`
- [x] LangChain `ChatGroq`
- [x] OpenAI ↔ Groq provider abstraction
- [x] LangChain `.invoke()` model calls
- [x] LangChain `bind_tools()` architecture

### Phase 2 — RAG Quality

- [ ] Token-aware chunking
- [ ] Structure-aware PDF parsing
- [ ] Metadata filtering
- [ ] Similarity thresholds
- [ ] Hybrid retrieval
- [ ] Cross-encoder reranking
- [ ] Better source citations
- [ ] Retrieval evaluation dataset
- [ ] Precision@k / Recall@k evaluation

### Phase 3 — Agentic Architecture

- [ ] Robust multi-step tool execution
- [ ] Explicit tool-state tracking
- [ ] Retry/fallback handling
- [ ] Tool error recovery
- [ ] Query rewriting
- [ ] Conversation-aware retrieval
- [ ] LangGraph orchestration
- [ ] Human-in-the-loop workflows

### Phase 4 — Production Readiness

- [ ] RAGAS evaluation
- [ ] Observability/tracing
- [ ] Latency monitoring
- [ ] Token/cost monitoring
- [ ] Structured logging
- [ ] Docker deployment
- [ ] Authentication
- [ ] Multi-user document isolation
- [ ] Cloud deployment

---

## Learning Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [LangChain OpenAI Integration](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [LangChain Groq Integration](https://docs.langchain.com/oss/python/integrations/chat/groq)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [uv Documentation](https://docs.astral.sh/uv/)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Jahid Md Hasan**

- GitHub: https://github.com/jmhasan1
- LinkedIn: https://www.linkedin.com/in/jahid-hasan-19jm/
- Portfolio: https://jmhasan1.github.io/Jahid_Portfolio/
- Email: jmhasan17@gmail.com

---

<!-- ## Project Goal

DocuMind AI is being developed as an interview-ready **Agentic RAG system** demonstrating practical skills in:

```text
RAG Engineering
├── Document ingestion
├── Chunking
├── Embeddings
├── Vector search
└── Grounded generation

Agentic AI
├── Tool calling
├── Multi-step execution
├── Tool selection
└── Conversation state

LLM Engineering
├── LangChain
├── OpenAI
├── Groq
├── Provider abstraction
└── Standardized tool interfaces

Application Engineering
├── Gradio
├── Local persistence
├── Environment management
└── Reproducible dependencies with uv
```

The goal is not merely to build a chatbot, but to demonstrate a modular RAG + agent system where retrieval, tools, LLM providers, and the UI can evolve independently. -->

