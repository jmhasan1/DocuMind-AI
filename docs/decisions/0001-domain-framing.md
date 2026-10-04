# 0001 — Domain Framing: Regulatory & Compliance Document Intelligence

**Status:** Accepted  
**Date:** 2026-08-21  
**Related:** [`docs/ROADMAP.md`](../ROADMAP.md)

## Context

DocuMind's roadmap includes retrieval engineering, agentic orchestration, grounding checks, evaluation, observability, security/governance, and document lifecycle management. Without a domain, these layers can become a generic collection of RAG features.

Two directions were considered:

1. **Horizontal framing** — a general adaptive/self-evaluating agentic RAG system.
2. **Vertical framing** — a document-intelligence system for a specific problem class.

## Decision

Adopt the vertical framing:

> **Regulatory & Compliance Document Intelligence**

The architecture remains evolutionary; the domain determines why each layer matters.

| Layer | Domain-driven purpose |
|---|---|
| Retrieval | Find the correct clause, definition, procedure, or requirement |
| Grounding/evidence checks | Reduce unsupported claims |
| Evaluation | Make answer quality defensible |
| Security/governance | Document-level access control and auditability |
| Document lifecycle | Prevent stale/superseded documents from silently producing answers |
| Cross-document reasoning | Connect related clauses, SOPs, and regulatory material |

## Rationale

- A concrete domain makes architecture easier for reviewers to evaluate.
- The framing aligns with prior QMS/compliance-oriented work.
- Public regulatory/QMS-style material can support development without proprietary data.
- The domain makes provenance, evaluation, governance, and lifecycle management first-class requirements.

## v0.1.0 scope

This is a positioning and architectural decision, not a claim that the complete compliance feature set exists.

v0.1.0 establishes:

- PDF ingestion
- local embeddings
- ChromaDB dense retrieval
- document-scoped retrieval
- agent/tool routing
- reproducible retrieval evaluation

Future releases progressively implement and evaluate the domain-driven requirements.

## Consequence

Future architecture decisions should answer:

> **What concrete document-intelligence problem does this capability solve, and how will we measure it?**

This keeps the roadmap from becoming a feature checklist.