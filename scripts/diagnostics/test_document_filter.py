from rag_core import retrieve_chunks

DOCUMENT_ID = "99f9c01da0ebd7bf"

results = retrieve_chunks(
    "What architecture does the paper propose?",
    top_k=5,
    document_id=DOCUMENT_ID,
)

print("=" * 72)
print("Document-Scoped Retrieval Test")
print("=" * 72)

print(f"Requested document: {DOCUMENT_ID}")
print(f"Results returned: {len(results)}")
print()

for rank, result in enumerate(results, start=1):
    metadata = result.get("metadata") or {}

    print(f"Rank {rank}")
    print(f"Filename: {metadata.get('filename')}")
    print(f"Document ID: {metadata.get('document_id')}")
    print(f"Chunk: {metadata.get('chunk_index')}")
    print(f"Distance: {result.get('distance')}")
    print(f"Text: {' '.join(result['text'][:200].split())}")
    print("-" * 72)