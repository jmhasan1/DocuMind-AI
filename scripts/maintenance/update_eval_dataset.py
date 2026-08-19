import json
from pathlib import Path

DATASET_PATH = Path("eval/dataset.jsonl")
DOCUMENT_ID = "99f9c01da0ebd7bf"

lines = []

with DATASET_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        item = json.loads(line)
        item["document_id"] = DOCUMENT_ID
        lines.append(item)

with DATASET_PATH.open("w", encoding="utf-8") as file:
    for item in lines:
        file.write(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )

print(f"Updated {len(lines)} evaluation records.")
print(f"Document ID: {DOCUMENT_ID}")