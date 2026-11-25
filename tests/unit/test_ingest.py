
"""Unit tests for ingestion and BM25 cache integration."""

from importlib import import_module
from unittest.mock import Mock

import pytest


@pytest.fixture
def ingest_module(monkeypatch):
    """Load ingestion with mocked external dependencies."""

    ingest = import_module("ingest")
    sparse = import_module("retrieval.sparse")

    mock_collection = Mock()
    mock_embed_model = Mock()

    mock_embed_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3]
    ]

    monkeypatch.setattr(ingest, "collection", mock_collection)
    monkeypatch.setattr(ingest, "embed_model", mock_embed_model)

    sparse.invalidate_bm25_cache()

    yield ingest, sparse, mock_collection

    sparse.invalidate_bm25_cache()


def test_successful_ingestion_invalidates_bm25_cache(
    ingest_module, monkeypatch, tmp_path
):
    ingest, sparse, collection = ingest_module

    # Avoid reading or hashing an actual PDF.
    monkeypatch.setattr(ingest, "_document_id", lambda path: "doc_a")
    monkeypatch.setattr(ingest, "_file_sha256", lambda path: "file_hash")
    monkeypatch.setattr(
        ingest,
        "load_pdf_pages",
        lambda path: [{"page_number": 1, "text": "Neural networks"}],
    )

    # Create a cached index before ingestion.
    sparse._bm25_cache["doc_a"] = Mock()

    report = ingest.ingest_document(tmp_path / "sample.pdf")

    assert report["status"] == "success"
    assert report["document_id"] == "doc_a"
    collection.upsert.assert_called_once()
    assert sparse._bm25_cache == {}


def test_failed_upsert_does_not_invalidate_bm25_cache(
    ingest_module, monkeypatch, tmp_path
):
    ingest, sparse, collection = ingest_module

    monkeypatch.setattr(ingest, "_document_id", lambda path: "doc_a")
    monkeypatch.setattr(ingest, "_file_sha256", lambda path: "file_hash")
    monkeypatch.setattr(
        ingest,
        "load_pdf_pages",
        lambda path: [{"page_number": 1, "text": "Neural networks"}],
    )

    cached_index = Mock()
    sparse._bm25_cache["doc_a"] = cached_index

    collection.upsert.side_effect = RuntimeError("Chroma upsert failed")

    with pytest.raises(RuntimeError, match="Chroma upsert failed"):
        ingest.ingest_document(tmp_path / "sample.pdf")

    assert sparse._bm25_cache["doc_a"] is cached_index
