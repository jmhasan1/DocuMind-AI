from collections import Counter

from rag_core import collection


def main():
    print("=" * 72)
    print("DocuMind ChromaDB Document Inventory")
    print("=" * 72)

    total = collection.count()

    print(f"Collection : {collection.name}")
    print(f"Total chunks: {total}")
    print()

    if total == 0:
        print("WARNING: Chroma collection is empty.")
        return

    data = collection.get(
        include=["metadatas"]
    )

    metadatas = data.get("metadatas", [])

    filenames = Counter()
    document_ids = Counter()
    source_types = Counter()

    for metadata in metadatas:
        if not metadata:
            continue

        filenames[metadata.get("filename", "<missing>")] += 1
        document_ids[metadata.get("document_id", "<missing>")] += 1
        source_types[metadata.get("source_type", "<missing>")] += 1

    print("Documents")
    print("-" * 72)

    for filename, count in filenames.most_common():
        print(f"{filename:<50} {count:>6} chunks")

    print()
    print("Document IDs")
    print("-" * 72)

    for document_id, count in document_ids.most_common():
        print(f"{document_id:<50} {count:>6} chunks")

    print()
    print("Source types")
    print("-" * 72)

    for source_type, count in source_types.most_common():
        print(f"{source_type:<50} {count:>6}")

    print()
    print("Metadata fields")
    print("-" * 72)

    all_fields = set()

    for metadata in metadatas:
        if metadata:
            all_fields.update(metadata.keys())

    for field in sorted(all_fields):
        print(field)


if __name__ == "__main__":
    main()