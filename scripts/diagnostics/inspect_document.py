from rag_core import collection

DOCUMENT_ID = "99f9c01da0ebd7bf"

result = collection.get(
    where={"document_id": DOCUMENT_ID},
    include=["documents", "metadatas"],
)

documents = result.get("documents", [])
metadatas = result.get("metadatas", [])

print("=" * 72)
print("Target Document Inspection")
print("=" * 72)
print(f"Document ID: {DOCUMENT_ID}")
print(f"Chunks: {len(documents)}")
print()

for metadata, document in zip(metadatas, documents):
    chunk_index = metadata.get("chunk_index", "?")
    filename = metadata.get("filename", "?")

    preview = " ".join(document[:500].split())

    print(f"Chunk {chunk_index}")
    print(f"File: {filename}")
    print(f"Text: {preview}")
    print("-" * 72)