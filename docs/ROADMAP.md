# DocuMind AI Roadmap

**Documentation snapshot:** 25 November 2026
**Product framing:** Regulatory & Compliance Document Intelligence
**Frozen release:** `v0.1.0 — Evaluated RAG Foundation`
**Current workstream:** Sprint 1 — Hybrid Retrieval
**Target release:** `v0.2.0 — Hybrid Retrieval`

DocuMind is being developed as a local-first document-intelligence system. Each retrieval or reasoning upgrade must be evaluated against the preceding frozen baseline. This roadmap distinguishes completed foundations, current implementation, pending work, and future targets.

## Current status at a glance

| Area | Status |
|---|---|
| v0.1 dense retrieval baseline | Frozen |
| Document-scoped retrieval | Implemented and regression-tested |
| Dense retriever module extraction | Implemented |
| `rank-bm25` dependency | Added |
| BM25 sparse retriever | Implemented and unit-tested |
| BM25 index lifecycle and cache invalidation | Implemented and tested |
| Document isolation in sparse retrieval | Covered by unit tests |
| Reciprocal Rank Fusion | Pending |
| Unified retrieval pipeline and mode selector | Pending |
| Dense/BM25/hybrid comparative benchmark | Pending |
| v0.2 experiment artifact | Pending |
| Cross-encoder reranking | Planned for v0.3.0 |
| Adaptive/self-correcting/agentic RAG | Future releases |

**Latest reported automated test result:** 31 passed, 1 non-blocking AnyIO deprecation warning.

The implementation statuses above reflect source code and test results reviewed during Sprint 1. They do not imply that the complete hybrid retrieval pipeline or comparative benchmark is finished.

## Release sequence

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

---

## v0.1.0 — Evaluated RAG Foundation

**Status: Frozen baseline**

### Scope

- PDF ingestion with PyMuPDF.
- Word-based chunking: 500 words / 50-word overlap.
- `all-MiniLM-L6-v2` embeddings.
- Persistent ChromaDB storage.
- Dense retrieval.
- Document-scoped retrieval.
- Agent/tool routing: RAG search, calculator, and document summary.
- OpenAI/Groq provider abstraction.
- Frozen 15-query retrieval benchmark.

### Frozen retrieval results

| Metric | Result |
|---|---:|
| Hit@1 | 66.7% (10/15) |
| Hit@3 | 86.7% (13/15) |
| Hit@5 | 93.3% (14/15) |
| MRR | 0.783 |

Latency is run-specific. Historical notes report different measurements; use the baseline artifact for the exact recorded metrics and document environment, warm-up, and repetitions in future comparisons.

**Immutable artifact:** `eval/results/v0.1_dense_baseline.json`
**Historical tag:** `v0.1-dense-baseline`

### Sprint 0 — Document isolation and test foundation

Reported completed and merged to `main`:

- Propagate active `document_id` from Gradio session state through agent and retrieval.
- Apply Chroma metadata filtering at retrieval time.
- Add pytest infrastructure and document-isolation regression coverage.
- Historical test result: 14 passed, 1 AnyIO deprecation warning.
- Historical manual two-document isolation scenario passed.

Sprint 0 is historical work. Its original test result is retained here and is not replaced by later Sprint 1 test counts.

---

## v0.2.0 — Hybrid Retrieval

**Status: In progress — Sprint 1**

### Objective

Test whether BM25 improves exact-match and domain-specific retrieval when combined with dense semantic retrieval, without changing benchmark semantics.

### Implemented work

- [x] Extract dense retrieval into `retrieval/dense.py`.
- [x] Preserve `rag_core.retrieve_chunks()` as a backward-compatible dense retrieval wrapper.
- [x] Add `rank-bm25` dependency.
- [x] Implement BM25 sparse retrieval in `retrieval/sparse.py`.
- [x] Use ChromaDB as the canonical persistent source of document chunks.
- [x] Implement lazy, in-memory BM25 index construction.
- [x] Cache indexes independently by document scope, including the unfiltered scope.
- [x] Implement cache invalidation after successful ingestion upsert.
- [x] Preserve document metadata and chunk provenance in sparse results.
- [x] Add sparse retrieval unit tests.
- [x] Add ingestion cache-invalidation tests.
- [x] Verify the reported full test suite: 31 passed, 1 warning.

### Remaining Sprint 1 work

- [ ] Implement Reciprocal Rank Fusion (RRF).
- [ ] Define and implement the unified retrieval pipeline.
- [ ] Add retrieval modes: `vector`, `bm25`, and `hybrid`.
- [ ] Preserve document isolation across all retrieval modes.
- [ ] Add fusion tests, including duplicate handling and deterministic ranking.
- [ ] Validate metadata and provenance through the complete retrieval pipeline.
- [ ] Compare dense-only, BM25-only, and hybrid RRF using the same 15-query dataset.
- [ ] Record Hit@1, Hit@3, Hit@5, MRR, and latency.
- [ ] Save a reproducible v0.2 experiment artifact.
- [ ] Update retrieval architecture documentation with measured behavior.
- [ ] Complete relevant manual validation and close Sprint 1.

### Current implementation decisions

| Concern | Current decision |
|---|---|
| Dense retrieval | Preserve as baseline/control |
| Sparse retrieval | BM25 using `rank-bm25` and `BM25Okapi` |
| Corpus source | Existing ChromaDB chunks |
| Tokenization | Lowercase word tokens using `\w+` |
| BM25 candidate depth | 10 |
| Dense candidate depth | 10 |
| Final result count | Default top 3 |
| RRF constant | `k=60` |
| RRF weighting | Equal weights initially |
| BM25 cache | Lazy, in-memory, per document scope |
| Cache invalidation | After successful ingestion upsert |
| Reranking | Excluded from Sprint 1 |

These are initial implementation parameters, not claims of optimal performance. The experiment must record the actual configuration used.

### Intended retrieval experiment

```text
Question ──┬── Dense retriever ── top-N ──┐
           │                              │
           └── BM25 retriever ─── top-N ──┤
                                          ↓
                                     RRF fusion
                                          ↓
                                    Final top-K
```

The diagram represents the intended architecture. RRF and the unified retrieval pipeline remain pending until implemented and tested.

### Evaluation requirements

- Preserve the dense-only control.
- Keep dataset, corpus, chunking, and relevance semantics unchanged.
- Apply document scope consistently before ranking.
- Report Hit@1, Hit@3, Hit@5, MRR, and latency.
- Save per-query ranks and configuration metadata.
- Record the code revision and environment.
- Do not populate metrics with expected or illustrative values.
- Do not overwrite the v0.1 artifact.

### Sprint 1 exit criteria

Sprint 1 can close when:

1. RRF is implemented and covered by automated tests.
2. The unified retrieval pipeline supports `vector`, `bm25`, and `hybrid`.
3. Document isolation is verified across all modes.
4. Metadata and provenance are preserved.
5. The full automated test suite passes.
6. The three retrieval modes are compared using the unchanged evaluation dataset.
7. A reproducible v0.2 experiment artifact is saved.
8. Documentation, README, and private Sprint 1 notes are updated.
9. Changes are reviewed and committed before the sprint is considered closed.

Hybrid retrieval is eligible for adoption only after implementation and evaluation demonstrate a defensible quality/latency trade-off.

---

## v0.3.0 — Reranking & Retrieval Evaluation

### Objective

Evaluate whether a cross-encoder improves precision near the top of the result list.

### Planned work

- Add cross-encoder reranking over fused candidates.
- Compare dense-only, BM25-only, hybrid RRF, and hybrid-plus-reranker.
- Record reranker model, candidate depth, latency, and resource usage.
- Expand from lexical phrase checks toward human-judged relevance.
- Inspect failure cases, not only aggregate scores.

### Exit criteria

Adopt reranking only if its measurable quality gain justifies its latency and local hardware cost.

---

## v0.4.0 — Adaptive RAG

### Objective

Select retrieval strategy and candidate depth based on query characteristics.

### Planned work

- Identify query classes, such as exact clause/identifier lookup versus conceptual questions.
- Define deterministic routing features before introducing an LLM router.
- Compare fixed hybrid retrieval with adaptive routing.
- Measure routing accuracy, retrieval quality, latency, and failure behavior.

### Exit criteria

Adaptive routing must outperform or provide a justified operational benefit over a fixed strategy.

---

## v0.5.0 — Self-Correction / Grounding

### Objective

Detect insufficient evidence and reduce unsupported answers.

### Planned work

- Evidence sufficiency checks.
- Bounded query rewrite/retry.
- Grounded-answer validation.
- Explicit abstention when evidence is insufficient.
- Tests for unanswerable and adversarial questions.
- Citation/source consistency checks.

### Exit criteria

Demonstrate improved grounded-answer quality without uncontrolled retries or unacceptable latency.

---

## v0.6.0 — Agentic RAG V2

### Objective

Make tool orchestration more predictable and recoverable.

### Planned work

- Structured tool inputs/outputs.
- Bounded agent loops.
- Explicit tool failure states.
- Predictable state transitions.
- Trusted document scope and provenance carried through every tool.
- Tests for tool-selection failures and retrieval bypass.

### Exit criteria

Agentic behavior must be observable, bounded, testable, and more useful than a simpler RAG flow for the target tasks.

---

## v0.7.0 — Multi-Agent RAG

### Objective

Evaluate specialized agents only if single-agent experiments show a real limitation.

### Potential roles

- Retrieval specialist.
- Research/synthesis specialist.
- Evidence/critic specialist.
- Orchestrator.

### Evaluation

Compare conventional RAG, single-agent RAG, and multi-agent RAG on the same task set. Account for latency, cost, coordination failures, and evidence consistency.

### Gate

Do not implement multi-agent orchestration solely for architectural novelty. Proceed only when specialization has an evidence-backed purpose.

---

## v0.8.0 — Evaluation & Observability Platform

### Planned capabilities

- Versioned datasets and judgments.
- Retrieval regression tests.
- Semantic/graded relevance.
- Answer relevance.
- Faithfulness/groundedness.
- Citation correctness.
- Adversarial and unanswerable queries.
- Latency and provider-cost measurement.
- Experiment comparison and reproducible artifacts.
- Traceability from question to retrieved evidence and generated answer.

This release should consolidate evaluation capabilities introduced in earlier releases rather than postpone all evaluation until v0.8.

---

## v0.9.0 — Production Hardening

### Potential areas

- API/service boundary.
- Containerization and deployment documentation.
- Authentication and authorization.
- Document-level access control.
- Audit logging.
- Document lifecycle, versioning, and supersession.
- Operational observability.
- Security testing.
- Configuration and secret management.
- Backup/recovery and data-retention policy.

### Gate

Production hardening requires explicit threat modeling and lifecycle requirements. A local prototype is not a production compliance platform.

---

## v1.0.0 — Production-Oriented Document Intelligence

A coherent system with:

- Measurable retrieval and grounded-answer quality.
- Reliable document isolation.
- Traceable source provenance and citation correctness.
- Document lifecycle/version handling.
- Justified and bounded agentic behavior.
- Regression evaluation and observability.
- Appropriate access control, auditability, and security.
- Documented deployment and operational constraints.

The v1.0 label is a target, not a claim of regulatory certification or legal correctness.

---

## Cross-release engineering priorities

These are important foundational concerns and should be scheduled when they block correctness, rather than treated as cosmetic cleanup:

1. **Page provenance:** carry page numbers through chunking; test citation metadata. Current chunk metadata does not yet provide `page_number`, so citations remain filename-based.
2. **Re-ingestion lifecycle:** remove or reconcile stale chunks when chunk layout changes.
3. **Document scope:** enforce trusted scope in every retrieval mode.
4. **Evaluation quality:** move beyond lexical phrase matching with judged relevance.
5. **Agent behavior:** test retrieval-tool bypass and bounded failure handling.
6. **Scanned PDFs:** assess OCR separately, with accuracy and hardware measurements.
7. **Local resource profile:** measure model memory, indexing time, retrieval latency, and reranker cost.

These priorities do not expand Sprint 1 scope unless a specific item is required to satisfy its exit criteria.

## Release discipline

- Historical evaluation artifacts are immutable.
- Architecture changes require experiments.
- Future capabilities are not documented as shipped.
- Do not change chunking, embedding model, corpus, and evaluation semantics simultaneously in a retrieval experiment.
- Record the code revision, dataset/corpus identity, parameters, environment, and per-query results.
- Keep the system practical on modest hardware.
- Prefer the simplest architecture that demonstrably solves the target document-intelligence problem.
- Complete sprint validation, documentation, README updates, private notes, and Git review before starting the next sprint.
- Maintain Codex handover documentation incrementally; finalize it at the end of v0.2.0.
