from importlib import import_module
from unittest.mock import Mock
import sys
import types

import pytest


@pytest.fixture
def agent_module(monkeypatch):
    fake_llm_factory = types.ModuleType("llm_factory")
    fake_llm_factory.get_llm = Mock()

    fake_rag_core = types.ModuleType("rag_core")
    fake_rag_core.rag_answer = Mock(return_value="Mocked RAG answer")

    monkeypatch.setitem(sys.modules, "llm_factory", fake_llm_factory)
    monkeypatch.setitem(sys.modules, "rag_core", fake_rag_core)

    agent = import_module("agent")

    monkeypatch.setattr(agent, "rag_answer", fake_rag_core.rag_answer)

    return agent, fake_rag_core.rag_answer


def test_rag_search_passes_document_id_to_rag_answer(agent_module):
    agent, mock_rag_answer = agent_module

    tools = agent._build_tools(
        provider="groq",
        chat_history=[],
        document_id="doc_123",
    )

    rag_search_tool = next(
        tool for tool in tools if tool.name == "rag_search"
    )

    rag_search_tool.invoke({"query": "What does this document say?"})

    mock_rag_answer.assert_called_once_with(
        "What does this document say?",
        chat_history=[],
        provider="groq",
        document_id="doc_123",
    )

def test_summarize_doc_passes_document_id_to_rag_answer(agent_module):
    agent, mock_rag_answer = agent_module

    tools = agent._build_tools(
        provider="groq",
        chat_history=[],
        document_id="doc_123",
    )

    summarize_tool = next(
        tool for tool in tools if tool.name == "summarize_doc"
    )

    summarize_tool.invoke({"topic": "financial reporting"})

    mock_rag_answer.assert_called_once_with(
        "Summarize everything about: financial reporting",
        chat_history=[],
        provider="groq",
        document_id="doc_123",
    )

def test_agent_loop_passes_document_id_to_build_tools(agent_module, monkeypatch):
    agent, _ = agent_module

    mock_build_tools = Mock(return_value=[])
    monkeypatch.setattr(agent, "_build_tools", mock_build_tools)

    mock_response = Mock()
    mock_response.tool_calls = []
    mock_response.content = "Hello"

    mock_llm_with_tools = Mock()
    mock_llm_with_tools.invoke.return_value = mock_response

    mock_llm = Mock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    monkeypatch.setattr(agent, "get_llm", Mock(return_value=mock_llm))

    agent.agent_loop(
        user_message="Hello",
        chat_history=[],
        provider="groq",
        document_id="doc_123",
    )

    mock_build_tools.assert_called_once_with(
        "groq",
        [],
        document_id="doc_123",
    )