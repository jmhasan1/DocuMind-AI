"""Run a lightweight retrieval baseline for DocuMind.

This intentionally evaluates the CURRENT dense retriever before we add hybrid
retrieval or reranking. Relevance is weakly supervised by manually curated
phrases in eval/dataset.jsonl. This is a baseline, not a substitute for a
future judged semantic evaluation set.

Run from the project root:
    uv run python -m eval.run_retrieval_baseline
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

from rag_core import retrieve_chunks


DATASET_PATH = Path(__file__).with_name("dataset.jsonl")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(result: dict, phrases: list[str]) -> bool:
    text = normalize(result.get("text", ""))
    return any(normalize(phrase) in text for phrase in phrases)


def evaluate(k_values: tuple[int, ...] = (1, 3, 5)) -> None:
    dataset = load_dataset()
    if not dataset:
        raise RuntimeError("Evaluation dataset is empty.")

    print("DocuMind retrieval baseline")
    print("=" * 72)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Queries: {len(dataset)}")
    print(f"Retriever: current dense Chroma + MiniLM")
    document_ids = {
    item["document_id"]
    for item in dataset
    }
    print(f"Evaluation document IDs: {', '.join(sorted(document_ids))}")
    print()

    per_query: list[dict] = []
    total_latency = 0.0

    for item in dataset:
        start = time.perf_counter()
        document_id = item.get("document_id")
        if not document_id:
            raise ValueError(
                f"Evaluation item {item.get('id', '<unknown>')} "
                "is missing document_id."
            )
        results = retrieve_chunks(
            item["question"],
            top_k=max(k_values),
            document_id=item["document_id"],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        total_latency += latency_ms

        first_relevant_rank = None
        for rank, result in enumerate(results, start=1):
            if is_relevant(result, item["relevant_phrases"]):
                first_relevant_rank = rank
                break

        per_query.append(
            {
                "id": item["id"],
                "first_relevant_rank": first_relevant_rank,
                "latency_ms": latency_ms,
                "top_result": results[0]["text"][:120].replace("\n", " ") if results else "<no result>",
            }
        )

    print("Per-query results")
    print("-" * 72)
    print(f"{'ID':<5} {'Rank':<6} {'Latency':>10}  Top result")
    for row in per_query:
        rank = str(row["first_relevant_rank"] or "-")
        print(f"{row['id']:<5} {rank:<6} {row['latency_ms']:>8.1f} ms  {row['top_result']}")

    print()
    print("Aggregate metrics")
    print("-" * 72)
    for k in k_values:
        hits = sum(
            1 for row in per_query
            if row["first_relevant_rank"] is not None and row["first_relevant_rank"] <= k
        )
        print(f"Hit@{k}: {hits / len(per_query):.3f} ({hits}/{len(per_query)})")

    rr_values = [
        1 / row["first_relevant_rank"]
        if row["first_relevant_rank"] is not None
        else 0.0
        for row in per_query
    ]
    print(f"MRR:   {sum(rr_values) / len(rr_values):.3f}")
    print(f"Avg retrieval latency: {total_latency / len(per_query):.1f} ms")
    print()
    print("NOTE: This baseline uses curated lexical evidence phrases. Next we will")
    print("upgrade the dataset to judged relevance labels and add semantic/generation metrics.")


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Make sure the document is ingested before running the baseline.", file=sys.stderr)
        raise
