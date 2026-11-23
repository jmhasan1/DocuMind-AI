from importlib import import_module
from unittest.mock import Mock
import sys
import types

import pytest


@pytest.fixture
def rag_core_module(monkeypatch):
    fake_ingest = types.ModuleType("ingest")
    fake_ingest.collection = Mock()
    fake_ingest.embed_model = Mock()

    fake_llm_factory = types.ModuleType("llm_factory")
    fake_llm_factory.get_llm = Mock()

    monkeypatch.setitem(sys.modules, "ingest", fake_ingest)
    monkeypatch.setitem(sys.modules, "llm_factory", fake_llm_factory)

    rag_core = import_module("rag_core")

    return rag_core


def test_rag_answer_calls_retrieval(rag_core_module, monkeypatch):
    mock_retrieve = Mock(return_value=[
        {
            "text": "The regulation requires annual reporting.",
            "metadata": {"filename": "regulation.pdf"},
            "distance": 0.1,
        }
    ])

    monkeypatch.setattr(rag_core_module, "retrieve_chunks", mock_retrieve)

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Annual reporting is required.")

    monkeypatch.setattr(
        rag_core_module,
        "get_llm",
        Mock(return_value=mock_llm),
    )

    result = rag_core_module.rag_answer("What is required?")

    assert result == "Annual reporting is required."
    mock_retrieve.assert_called_once()
    mock_llm.invoke.assert_called_once()


def test_rag_answer_returns_message_when_no_chunks(
    rag_core_module, monkeypatch
):
    mock_retrieve = Mock(return_value=[])

    monkeypatch.setattr(rag_core_module, "retrieve_chunks", mock_retrieve)

    result = rag_core_module.rag_answer("What is in the document?")

    assert "couldn't find any indexed document content" in result.lower()

def test_rag_answer_passes_document_id_to_retrieval(
    rag_core_module, monkeypatch
):
    mock_retrieve = Mock(return_value=[
        {
            "text": "Document-specific content",
            "metadata": {"document_id": "doc_123"},
            "distance": 0.1,
        }
    ])

    monkeypatch.setattr(rag_core_module, "retrieve_chunks", mock_retrieve)

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Document-specific answer.")

    monkeypatch.setattr(
        rag_core_module,
        "get_llm",
        Mock(return_value=mock_llm),
    )

    rag_core_module.rag_answer(
        "What does this document say?",
        document_id="doc_123",
    )

    mock_retrieve.assert_called_once_with(
        "What does this document say?",
        top_k=3,
        document_id="doc_123",
    )