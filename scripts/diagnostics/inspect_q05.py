from rag_core import retrieve_chunks


DOCUMENT_ID = "99f9c01da0ebd7bf"

QUESTION = "How many NVIDIA P100 GPUs were used to train the models?"


def main():
    results = retrieve_chunks(
        QUESTION,
        top_k=14,
        document_id=DOCUMENT_ID,
    )

    print("=" * 80)
    print("Q05 Retrieval Diagnostic")
    print("=" * 80)

    print(f"Question: {QUESTION}")
    print(f"Document ID: {DOCUMENT_ID}")
    print(f"Results returned: {len(results)}")
    print()

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}

        chunk_index = metadata.get("chunk_index", "?")
        distance = result.get("distance")

        text = " ".join(result.get("text", "").split())

        print(f"RANK {rank}")
        print(f"Chunk: {chunk_index}")
        print(f"Distance: {distance}")
        print(f"Text: {text}")
        print("-" * 80)


if __name__ == "__main__":
    main()