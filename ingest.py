"""Document ingestion and vector-store access for DocuMind AI.

Phase 0 goals:
- No ingestion side effects at import time.
- Deterministic document/chunk identifiers.
- Preserve document and chunk provenance in Chroma metadata.
- Keep the embedding model and vector store reusable by the rest of the app.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
import fitz
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


# These objects are intentionally initialized at import time because they are
# reusable infrastructure, not document-ingestion side effects. No PDF is read
# and no vectors are written until ingest_document() is explicitly called.
embed_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
collection = client.get_or_create_collection(DEFAULT_COLLECTION_NAME)


def load_pdf(path: str | Path) -> str:
    """Extract text from all pages of a PDF."""
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")

    with fitz.open(pdf_path) as doc:
        pages = [page.get_text("text") for page in doc]

    return "\n\n".join(page.strip() for page in pages if page.strip())


def load_pdf_pages(path: str | Path) -> list[dict[str, Any]]:
    """Extract page-level text while preserving page numbers for provenance."""
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")

    with fitz.open(pdf_path) as doc:
        return [
            {
                "page_number": page_number,
                "text": page.get_text("text").strip(),
            }
            for page_number, page in enumerate(doc, start=1)
            if page.get_text("text").strip()
        ]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word-based chunks.

    This remains intentionally simple in Phase 0. We will replace/benchmark
    chunking strategies in the retrieval-quality phase.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
    ]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_id(path: str | Path) -> str:
    """Create a stable document ID from file contents, not its local path."""
    return _file_sha256(path)[:16]


def _chunk_id(document_id: str, chunk_index: int, chunk_text_value: str) -> str:
    """Create a stable chunk ID that survives repeated ingestion."""
    chunk_hash = _sha256_text(chunk_text_value)[:12]
    return f"{document_id}_chunk_{chunk_index}_{chunk_hash}"


def ingest_document(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Parse, chunk, embed, and index one PDF explicitly.

    Returns a small ingestion report suitable for the UI and tests.
    Re-ingesting an identical file is idempotent because chunk IDs are stable.
    """
    pdf_path = Path(path)
    document_id = _document_id(pdf_path)
    file_hash = _file_sha256(pdf_path)
    pages = load_pdf_pages(pdf_path)

    full_text = "\n\n".join(page["text"] for page in pages)
    chunks = chunk_text(
        full_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        return {
            "status": "empty",
            "document_id": document_id,
            "filename": pdf_path.name,
            "chunks_indexed": 0,
        }

    embeddings = embed_model.encode(chunks).tolist()
    ids = [
        _chunk_id(document_id, index, chunk)
        for index, chunk in enumerate(chunks)
    ]

    metadatas = [
        {
            "document_id": document_id,
            "filename": pdf_path.name,
            "file_hash": file_hash,
            "chunk_index": index,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "source_type": "pdf",
        }
        for index, _ in enumerate(chunks)
    ]

    # Chroma's upsert makes repeated ingestion of the same document safe.
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    return {
        "status": "success",
        "document_id": document_id,
        "filename": pdf_path.name,
        "file_hash": file_hash,
        "pages_with_text": len(pages),
        "chunks_indexed": len(chunks),
    }