# 0002 — Hybrid Retrieval (BM25 + Dense + Reranking)

**Status:** Accepted, pending real-corpus benchmark confirmation
**Date:** 2026-08-21
**Related:** [`docs/architecture/retrieval.md`](../architecture/retrieval.md),
[`docs/ROADMAP.md`](../ROADMAP.md) (v0.3)

## Context

v0.2's retrieval was dense-only (Chroma + `all-MiniLM-L6-v2`). Per the domain framing
established in [0001](0001-domain-framing.md), DocuMind targets compliance/QMS documents,
where questions frequently anchor on exact lexical content — clause numbers, defined
terms, form-field names — that dense embeddings are not specifically optimized to match.

## Decision

Add a BM25 sparse retrieval path (`rank_bm25`), fuse it with the existing dense path using
Reciprocal Rank Fusion, and add a cross-encoder reranking pass over the fused candidates.
Retrieval mode is configurable (`vector` / `bm25` / `hybrid` / `hybrid_rerank`) specifically
so the choice is benchmarked, not assumed.

## Alternatives Considered

- **Weighted score blend of BM25 and cosine similarity**, instead of RRF. Rejected — the
  two scores aren't on comparable scales, and a hand-tuned blend weight is an unvalidated
  free parameter that would need its own justification. RRF uses only rank order from each
  retriever, avoiding the scale problem entirely.
- **Dense-only with a larger/better embedding model** instead of adding a sparse path.
  Rejected as the primary fix — a stronger embedding model doesn't change dense retrieval's
  structural weakness on exact-match content (clause numbers, defined terms); BM25
  specifically targets that gap.
- **Skip reranking, ship hybrid RRF alone.** Considered but reranking is retained because
  its main value — precision at rank 1 — matters disproportionately for a system meant to
  cite a specific source, not just retrieve a plausible top-5.

## Consequences

- New dependency: `rank-bm25` (pure Python, no additional infrastructure).
- The BM25 index is rebuilt from the Chroma collection's contents per retrieval call
  rather than persisted separately. Acceptable at current corpus scale (single-digit
  documents); flagged in `retrieval.py` as something to revisit if corpus size grows
  enough to make that rebuild cost matter — not fixed preemptively.
- Ingestion (`ingest.py`) was changed to chunk per-page instead of per-document, so
  `page_number` is now actually populated in chunk metadata. This closes a latent gap:
  `rag_core.py` was already reading `metadata.get("page_number")` for citations, but it
  was never set, so citations silently degraded to filename-only.
- Detected tables are now indexed as separate atomic chunks (`content_type: "table"`),
  with a documented known limitation: table content is not yet excluded from the page's
  prose extraction, so it currently appears in both places. See
  `docs/architecture/retrieval.md` for why this wasn't fixed preemptively.

## Outcome — to be completed after running the benchmark locally

```bash
uv run python -m eval.run_retrieval_baseline
```

_This section intentionally left as a template. The retrieval pipeline was verified
end-to-end with a synthetic document and stubbed embedding/reranker models (this sandbox
has no live access to Hugging Face Hub to download the real models), confirming ingestion,
BM25 indexing, RRF fusion, and reranking all execute correctly — but the actual Hit@k/MRR
numbers require running against the real corpus and models locally._

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Avg latency |
|---|---|---|---|---|---|
| vector | _fill in_ | | | | |
| bm25 | | | | | |
| hybrid | | | | | |
| hybrid_rerank | | | | | |

**Selected configuration:** _fill in once the numbers are in — the roadmap's default
target is `hybrid_rerank`, but this table is what actually justifies that, not the plan._
