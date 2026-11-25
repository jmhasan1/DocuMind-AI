"""Retrieval + generation layer for DocuMind AI."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from retrieval.dense import retrieve_dense
from llm_factory import get_llm

# This function converts the UI's conversation history into LangChain message objects.
def _history_to_messages(chat_history):
    messages = []
    for turn in chat_history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def retrieve_chunks(
    query: str,
    top_k: int = 3,
    document_id: str | None = None,
) -> list[dict]:
    """Backward-compatible wrapper for dense retrieval."""

    return retrieve_dense(
        query=query,
        top_k=top_k,
        document_id=document_id,
    )

# The RAG pipeline
def rag_answer(
    query,
    chat_history=None,
    provider="groq",
    top_k=3,
    document_id=None,
    ):
    """Retrieve relevant chunks and generate a grounded answer."""
    retrieved = retrieve_chunks(
    query,
    top_k=top_k,
    document_id=document_id,
    )                               # Retrieve

    if not retrieved:
        return (
            "I couldn't find any indexed document content. "
            "Please upload and ingest a PDF first."
        )
    # Building contxt_parts 
    context_parts = []
    for index, item in enumerate(retrieved, start=1):
        metadata = item["metadata"] or {}
        filename = metadata.get("filename", "unknown document")
        page = metadata.get("page_number")
        page_label = f", page {page}" if page else ""
        context_parts.append(                                           # Building Source Text (LLM recieves explicit source boundaries)
            f"[Source {index} | {filename}{page_label}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)            # Combining the sources to produce the retrieved context.

    messages = [
        SystemMessage(                      # RAG System prompt establishes a grounding poicy ( main hallucintion control mechanism)
            content=(
                "You are the RAG answerer for DocuMind AI. Answer using the "
                "retrieved document context. If the answer is not supported "
                "by the context, say that the information was not found in "
                "the uploaded documents. Do not fabricate citations or facts."
            )
        )
    ]
    messages.extend(_history_to_messages(chat_history))
    messages.append(                                                # Add Retrieved Context + Current Question
        HumanMessage(
            content=(
                f"Retrieved document context:\n{context}\n\n"
                f"Current question: {query}\n\n"
                "Answer concisely and cite the retrieved passages as "
                "[Source 1], [Source 2], etc. when useful."
            )
        )
    )

    llm = get_llm(provider)
    response = llm.invoke(messages)
    return response.content
