# DocuMind AI Roadmap

**Current release:** `v0.1.0 — Evaluated RAG Foundation`

DocuMind is being developed as a **Regulatory & Compliance Document Intelligence** system. Each retrieval or reasoning upgrade must be evaluated against the preceding frozen baseline.

## Release sequence

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

## v0.1.0 — Evaluated RAG Foundation

Establish the clean, reproducible foundation:

- PDF ingestion
- provenance-aware metadata
- dense retrieval with `all-MiniLM-L6-v2` + ChromaDB
- document-scoped retrieval
- existing agent/tool routing
- OpenAI/Groq abstraction
- frozen 15-query benchmark
- Hit@1, Hit@3, Hit@5, MRR, latency

Frozen baseline:

- Hit@1: 66.7%
- Hit@3: 86.7%
- Hit@5: 93.3%
- MRR: 0.783
- Average retrieval latency: 24.4 ms

## v0.2.0 — Hybrid Retrieval

Test whether BM25 + dense retrieval improves exact-match/domain-specific retrieval.

```text
Query → Dense ─┐
               ├→ RRF → candidates
Query → BM25 ──┘
```

The dense path remains the control. Benchmark semantics remain unchanged.

## v0.3.0 — Reranking & Retrieval Evaluation

Test a cross-encoder over fused candidates. Compare dense-only, BM25-only, hybrid, and hybrid+rereanking configurations.

Begin expanding evaluation toward judged relevance.

## v0.4.0 — Adaptive RAG

Route queries between retrieval strategies and depths based on query characteristics. Evaluate quality against added latency and complexity.

## v0.5.0 — Self-Correction / Grounding

Add evidence sufficiency checks, query rewriting/retry, grounded-answer validation, and bounded abstention behavior. Demonstrate measurable benefit without uncontrolled loops.

## v0.6.0 — Agentic RAG V2

Improve planning/recovery while preserving current tools:

- structured tool execution
- bounded loops
- explicit failure states
- provenance
- predictable state transitions

## v0.7.0 — Multi-Agent RAG

Only proceed if experiments justify specialization. Potential roles include retrieval, research/synthesis, critic/evidence, and orchestration agents.

Compare naive RAG vs agentic RAG vs multi-agent RAG.

## v0.8.0 — Evaluation & Observability Platform

Build reusable regression infrastructure:

- versioned datasets
- retrieval regression tests
- semantic relevance judgments
- answer relevance
- faithfulness/groundedness
- citation correctness
- adversarial/unanswerable questions
- latency/cost measurements
- experiment comparison

## v0.9.0 — Production Hardening

Potential areas:

- API/service boundary
- containerization
- authentication/authorization
- document-level access control
- audit logging
- document lifecycle
- observability
- security testing
- deployment documentation

## v1.0.0 — Production-Oriented Document Intelligence

A coherent system with measurable retrieval and grounded-answer quality, provenance, document isolation/lifecycle handling, justified agentic behavior, evaluation/regression infrastructure, observability, and appropriate security/governance.

## Release discipline

- Historical evaluation artifacts are immutable.
- Architecture changes require experiments.
- Future capabilities are not documented as shipped.
- The system remains practical on modest hardware.