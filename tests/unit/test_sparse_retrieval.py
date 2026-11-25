
"""Unit tests for BM25 sparse retrieval."""

from importlib import import_module
from unittest.mock import Mock

import pytest

@pytest.fixture
def sparse_module(monkeypatch):
    """Provide an isolated BM25 retriever and mock Chroma collection."""

    sparse = import_module("retrieval.sparse")
    sparse.invalidate_bm25_cache()

    mock_collection = Mock()
    mock_collection.count.return_value = 1

    monkeypatch.setattr(sparse, "collection", mock_collection)

    yield sparse, mock_collection

    sparse.invalidate_bm25_cache()



def make_chroma_records(records):
    """Build a Chroma-compatible get() response."""

    return {
        "ids": [record[0] for record in records],
        "documents": [record[1] for record in records],
        "metadatas": [record[2] for record in records],
    }


def test_empty_query_returns_empty_list(sparse_module):
    sparse, collection = sparse_module

    collection.count.return_value = 0
    assert sparse.retrieve_bm25("   ") == []
    collection.get.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_value_error(sparse_module, top_k):
    sparse, _ = sparse_module

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        sparse.retrieve_bm25("neural network", top_k=top_k)


def test_blank_document_id_raises_value_error(sparse_module):
    sparse, _ = sparse_module

    with pytest.raises(ValueError, match="document_id must be non-empty"):
        sparse.retrieve_bm25("neural network", document_id="   ")


def test_tokenize_lowercases_and_splits_text(sparse_module):
    sparse, _ = sparse_module

    assert sparse.tokenize("Neural NETWORK, Retrieval!") == [
        "neural", "network", "retrieval"
    ]


def test_bm25_returns_matching_chunks(sparse_module):
    sparse, collection = sparse_module
    collection.count.return_value = 2

    collection.get.return_value = make_chroma_records([
        (
            "chunk_a",
            "Neural networks are used in deep learning.",
            {"document_id": "doc_a", "chunk_index": 0},
        ),
        (
            "chunk_b",
            "Bananas contain potassium and fiber.",
            {"document_id": "doc_a", "chunk_index": 1},
        ),
    ])

    results = sparse.retrieve_bm25("neural networks", top_k=2)

    assert len(results) == 1
    assert results[0]["text"] == (
        "Neural networks are used in deep learning."
    )
    assert results[0]["metadata"]["document_id"] == "doc_a"
    assert results[0]["distance"] is None
    assert isinstance(results[0]["score"], float)
    assert results[0]["retrieval_method"] == "bm25"


def test_document_id_filter_is_applied_before_indexing(sparse_module):
    sparse, collection = sparse_module

    collection.get.return_value = make_chroma_records([
        (
            "chunk_a",
            "Neural networks and deep learning.",
            {"document_id": "doc_a"},
        ),
    ])

    sparse.retrieve_bm25("neural", document_id="doc_a")

    collection.get.assert_called_once_with(
    where={"document_id": "doc_a"},
    limit=sparse.BM25_BATCH_SIZE,
    offset=0,
    include=["documents", "metadatas"],
    )


def test_unfiltered_retrieval_uses_no_document_filter(sparse_module):
    sparse, collection = sparse_module

    collection.get.return_value = make_chroma_records([
        (
            "chunk_a",
            "Neural networks and deep learning.",
            {"document_id": "doc_a"},
        ),
    ])

    sparse.retrieve_bm25("neural")

    collection.get.assert_called_once_with(
    where=None,
    limit=sparse.BM25_BATCH_SIZE,
    offset=0,
    include=["documents", "metadatas"],
    )


def test_cache_is_reused_for_repeated_queries(sparse_module):
    sparse, collection = sparse_module

    collection.get.return_value = make_chroma_records([
        (
            "chunk_a",
            "Neural networks and deep learning.",
            {"document_id": "doc_a"},
        ),
    ])

    sparse.retrieve_bm25("neural", document_id="doc_a")
    sparse.retrieve_bm25("networks", document_id="doc_a")

    assert collection.get.call_count == 1


def test_cache_is_separate_for_each_document(sparse_module):
    sparse, collection = sparse_module

    collection.get.side_effect = [
        make_chroma_records([
            ("a1", "Neural networks.", {"document_id": "doc_a"}),
        ]),
        make_chroma_records([
            ("b1", "Neural computation.", {"document_id": "doc_b"}),
        ]),
    ]

    result_a = sparse.retrieve_bm25("neural", document_id="doc_a")
    result_b = sparse.retrieve_bm25("neural", document_id="doc_b")

    assert collection.get.call_count == 2
    assert result_a[0]["metadata"]["document_id"] == "doc_a"
    assert result_b[0]["metadata"]["document_id"] == "doc_b"


def test_cache_invalidation_rebuilds_index(sparse_module):
    sparse, collection = sparse_module

    collection.get.return_value = make_chroma_records([
        (
            "chunk_a",
            "Neural networks and deep learning.",
            {"document_id": "doc_a"},
        ),
    ])

    sparse.retrieve_bm25("neural", document_id="doc_a")
    sparse.retrieve_bm25("neural", document_id="doc_a")

    assert collection.get.call_count == 1

    sparse.invalidate_bm25_cache()

    sparse.retrieve_bm25("neural", document_id="doc_a")

    assert collection.get.call_count == 2


def test_empty_collection_returns_empty_list(sparse_module):
    sparse, collection = sparse_module
    collection.count.return_value = 0
    collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }

    assert sparse.retrieve_bm25("neural") == []


def test_non_matching_query_returns_empty_list(sparse_module):
    sparse, collection = sparse_module

    collection.get.return_value = make_chroma_records([
        (
            "chunk_a",
            "Neural networks and deep learning.",
            {"document_id": "doc_a"},
        ),
    ])

    assert sparse.retrieve_bm25("volcano") == []

def test_build_bm25_index_fetches_all_batches(monkeypatch):
    from retrieval import sparse

    monkeypatch.setattr(sparse.collection, "count", lambda **kwargs: 600)

    calls = []

    def fake_get(**kwargs):
        calls.append(kwargs)
        offset = kwargs["offset"]

        if offset == 0:
            return {
                "ids": [f"id-{i:03}" for i in range(500)],
                "documents": [f"document token {i}" for i in range(500)],
                "metadatas": [{} for _ in range(500)],
            }

        return {
            "ids": [f"id-{i:03}" for i in range(500, 600)],
            "documents": [f"document token {i}" for i in range(500, 600)],
            "metadatas": [{} for _ in range(100)],
        }

    monkeypatch.setattr(sparse.collection, "get", fake_get)

    index = sparse._build_bm25_index()

    assert index is not None
    assert len(index.documents) == 600
    assert len(calls) == 2
    assert calls[0]["offset"] == 0
    assert calls[1]["offset"] == 500

def test_empty_bm25_index_is_cached(monkeypatch):
    from retrieval import sparse

    sparse.invalidate_bm25_cache()

    count_calls = []

    def fake_count(**kwargs):
        count_calls.append(kwargs)
        return 0

    monkeypatch.setattr(sparse.collection, "count", fake_count)

    assert sparse._get_bm25_index("empty-document") is None
    assert sparse._get_bm25_index("empty-document") is None

    assert len(count_calls) == 1