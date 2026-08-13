"""Retrieval + generation layer for DocuMind AI."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ingest import collection, embed_model
from llm_factory import get_llm


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


def retrieve_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve document chunks with provenance and similarity distance.

    Chroma returns distances rather than normalized similarity scores. We keep
    the raw distance in Phase 0 and will define/benchmark scoring semantics in
    the retrieval-engineering phase.
    """
    if not query or not query.strip():
        return []
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for index, document in enumerate(documents):
        retrieved.append(
            {
                "text": document,
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    return retrieved


def rag_answer(query, chat_history=None, provider="groq", top_k=3):
    """Retrieve relevant chunks and generate a grounded answer."""
    retrieved = retrieve_chunks(query, top_k=top_k)

    if not retrieved:
        return (
            "I couldn't find any indexed document content. "
            "Please upload and ingest a PDF first."
        )

    context_parts = []
    for index, item in enumerate(retrieved, start=1):
        metadata = item["metadata"] or {}
        filename = metadata.get("filename", "unknown document")
        page = metadata.get("page_number")
        page_label = f", page {page}" if page else ""
        context_parts.append(
            f"[Source {index} | {filename}{page_label}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(
            content=(
                "You are the RAG answerer for DocuMind AI. Answer using the "
                "retrieved document context. If the answer is not supported "
                "by the context, say that the information was not found in "
                "the uploaded documents. Do not fabricate citations or facts."
            )
        )
    ]
    messages.extend(_history_to_messages(chat_history))
    messages.append(
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
