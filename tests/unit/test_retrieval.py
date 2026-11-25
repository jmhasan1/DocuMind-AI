from importlib import import_module
from unittest.mock import Mock
import sys
import types

import pytest


@pytest.fixture
def rag_core_module(monkeypatch):
    """Import rag_core without loading real ChromaDB or embedding models."""

    fake_ingest = types.ModuleType("ingest")
    fake_ingest.collection = Mock()
    fake_ingest.embed_model = Mock()

    fake_llm_factory = types.ModuleType("llm_factory")
    fake_llm_factory.get_llm = Mock()

    monkeypatch.setitem(sys.modules, "ingest", fake_ingest)
    monkeypatch.setitem(sys.modules, "llm_factory", fake_llm_factory)

    rag_core = import_module("rag_core")

    return rag_core


@pytest.fixture
def mock_retrieval_dependencies(rag_core_module, monkeypatch):
    """Provide isolated mocks for dense retrieval dependencies."""

    dense_module = import_module("retrieval.dense")

    mock_collection = Mock()
    mock_embed_model = Mock()

    monkeypatch.setattr(dense_module, "collection", mock_collection)
    monkeypatch.setattr(dense_module, "embed_model", mock_embed_model)

    return rag_core_module, mock_collection, mock_embed_model



def test_empty_query_returns_empty_list(mock_retrieval_dependencies):
    rag_core, _, mock_embed_model = mock_retrieval_dependencies

    result = rag_core.retrieve_chunks("   ")

    assert result == []
    mock_embed_model.encode.assert_not_called()


def test_invalid_top_k_raises_value_error(mock_retrieval_dependencies):
    rag_core, _, _ = mock_retrieval_dependencies

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        rag_core.retrieve_chunks("test query", top_k=0)


def test_document_id_filter_is_passed_to_chroma(mock_retrieval_dependencies):
    rag_core, mock_collection, mock_embed_model = mock_retrieval_dependencies

    mock_collection.query.return_value = {
        "documents": [["Relevant text"]],
        "metadatas": [[{"document_id": "doc_123"}]],
        "distances": [[0.25]],
    }

    mock_embed_model.encode.return_value = Mock(
        tolist=lambda: [[0.1, 0.2, 0.3]]
    )

    result = rag_core.retrieve_chunks(
        "test query",
        document_id="doc_123",
    )

    query_kwargs = mock_collection.query.call_args.kwargs

    assert query_kwargs["where"] == {"document_id": "doc_123"}
    assert result[0]["text"] == "Relevant text"
    assert result[0]["metadata"]["document_id"] == "doc_123"
    assert result[0]["distance"] == 0.25


def test_retrieval_returns_text_metadata_and_distance(
    mock_retrieval_dependencies,
):
    rag_core, mock_collection, mock_embed_model = mock_retrieval_dependencies

    mock_collection.query.return_value = {
        "documents": [["Chunk A", "Chunk B"]],
        "metadatas": [[{"source": "A"}, {"source": "B"}]],
        "distances": [[0.1, 0.2]],
    }

    mock_embed_model.encode.return_value = Mock(
        tolist=lambda: [[0.1, 0.2, 0.3]]
    )

    result = rag_core.retrieve_chunks("test query", top_k=2)

    assert len(result) == 2

    assert result[0] == {
        "text": "Chunk A",
        "metadata": {"source": "A"},
        "distance": 0.1,
    }

    assert result[1] == {
        "text": "Chunk B",
        "metadata": {"source": "B"},
        "distance": 0.2,
    }

    query_kwargs = mock_collection.query.call_args.kwargs
    assert query_kwargs["n_results"] == 2

def test_negative_top_k_raises_value_error(mock_retrieval_dependencies):
    rag_core, _, _ = mock_retrieval_dependencies

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        rag_core.retrieve_chunks("test query", top_k=-1)


def test_blank_document_id_raises_value_error(mock_retrieval_dependencies):
    rag_core, _, _ = mock_retrieval_dependencies

    with pytest.raises(ValueError, match="document_id must be non-empty"):
        rag_core.retrieve_chunks("test query", document_id="   ")


def test_retrieval_without_document_id_has_no_filter(
    mock_retrieval_dependencies,
):
    rag_core, mock_collection, mock_embed_model = mock_retrieval_dependencies

    mock_collection.query.return_value = {
        "documents": [["Chunk A"]],
        "metadatas": [[{"source": "A"}]],
        "distances": [[0.1]],
    }

    mock_embed_model.encode.return_value = Mock(
        tolist=lambda: [[0.1, 0.2, 0.3]]
    )

    rag_core.retrieve_chunks("test query")

    query_kwargs = mock_collection.query.call_args.kwargs

    assert "where" not in query_kwargs