"""Dense retrieval implementation for DocuMind AI."""

from __future__ import annotations

from ingest import collection, embed_model


def retrieve_dense(
    query: str,
    top_k: int = 3,
    document_id: str | None = None,
) -> list[dict]:
    """Retrieve document chunks using dense vector similarity."""

    if not query or not query.strip():
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if document_id is not None and not document_id.strip():
        raise ValueError("document_id must be non-empty when provided")

    query_embedding = embed_model.encode([query]).tolist()

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if document_id is not None:
        query_kwargs["where"] = {"document_id": document_id}

    results = collection.query(**query_kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for index, document in enumerate(documents):
        retrieved.append(
            {
                "text": document,
                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                ),
            }
        )

    return retrieved