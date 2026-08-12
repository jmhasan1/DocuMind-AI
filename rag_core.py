"""Retrieval + generation layer for DocuMind AI."""
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


def rag_answer(query, chat_history=None, provider="groq"):
    """Retrieve relevant chunks and generate a grounded answer."""
    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    documents = results.get("documents", [[]])
    chunks = documents[0] if documents else []

    if not chunks:
        return (
            "I couldn't find any indexed document content. "
            "Please upload and ingest a PDF first."
        )

    context = "\n\n".join(
        f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

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
