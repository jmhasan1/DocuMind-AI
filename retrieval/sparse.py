
"""Sparse BM25 retrieval implementation for DocuMind AI."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ingest import collection

BM25_BATCH_SIZE = 500


@dataclass
class BM25Index:
    """Cached BM25 index and its corresponding Chroma records."""

    documents: list[str]
    metadatas: list[dict]
    ids: list[str]
    index: BM25Okapi
    tokenized_documents: list[list[str]]


# None represents the complete collection.
# Empty scopes are cached as None until invalidation.
_bm25_cache: dict[str | None, BM25Index | None] = {}


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def invalidate_bm25_cache() -> None:
    """Clear all cached indexes after successful ingestion."""
    _bm25_cache.clear()


def _build_bm25_index(document_id: str | None = None) -> BM25Index | None:
    """Fetch the complete scoped Chroma corpus and build a BM25 index."""

    where = {"document_id": document_id} if document_id is not None else None
    total_count = collection.count(where=where)

    if total_count == 0:
        return None

    records = []

    for offset in range(0, total_count, BM25_BATCH_SIZE):
        results = collection.get(
            where=where,
            limit=BM25_BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        records.extend(
            zip(ids, documents, metadatas, strict=True)
        )

    # Deterministic ordering makes equal-score ranking reproducible.
    records.sort(key=lambda record: record[0])

    # Exclude chunks with no searchable tokens.
    searchable_records = []

    for chunk_id, document, metadata in records:
        tokens = tokenize(document)
        if tokens:
            searchable_records.append(
                (chunk_id, document, metadata or {}, tokens)
            )

    if not searchable_records:
        return None

    return BM25Index(
        ids=[record[0] for record in searchable_records],
        documents=[record[1] for record in searchable_records],
        metadatas=[record[2] for record in searchable_records],
        tokenized_documents=[record[3] for record in searchable_records],
        index=BM25Okapi(
            [record[3] for record in searchable_records]
        ),
    )


def _get_bm25_index(document_id: str | None = None) -> BM25Index | None:
    """Return a cached index or build it on first use."""

    if document_id not in _bm25_cache:
        _bm25_cache[document_id] = _build_bm25_index(document_id)

    return _bm25_cache[document_id]


def retrieve_bm25(
    query: str,
    top_k: int = 3,
    document_id: str | None = None,
) -> list[dict]:
    """Retrieve chunks using document-scoped BM25 lexical ranking."""

    if not query or not query.strip():
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if document_id is not None and not document_id.strip():
        raise ValueError("document_id must be non-empty when provided")

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    bm25_data = _get_bm25_index(document_id)

    if bm25_data is None:
        return []

    scores = bm25_data.index.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            bm25_data.ids[index],
        ),
    )

    results = []

    query_token_set = set(query_tokens)

    for index in ranked_indices:
        # Exclude chunks without an actual lexical match.
        if not query_token_set.intersection(
            bm25_data.tokenized_documents[index]
        ):
            continue

        results.append(
            {
                "text": bm25_data.documents[index],
                "metadata": bm25_data.metadatas[index],
                "distance": None,
                "score": float(scores[index]),
                "retrieval_method": "bm25",
            }
        )

        if len(results) >= top_k:
            break

    return results
