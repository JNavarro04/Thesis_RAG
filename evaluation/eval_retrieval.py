# eval_retrieval.py
#
# Retrieval evaluation for both chunking strategies.
# For each of the 150 QA questions, queries both ChromaDB collections
# and measures whether the correct chunk was retrieved.
#
# Correctness definition:
#   A retrieved chunk is correct if the question's source_entity
#   (e.g. "ZONE_02", "ITEM_04", "ROUTE_05") appears in the chunk's
#   entry_header. This matches the primary entry for that entity.
#   For disambiguation questions with source_entity "SECTION_8" or
#   "SECTION_0", the match is against the section header text.
#
# Metrics computed at k = 1, 3, 5:
#   Recall@k  — fraction of questions where a correct chunk appears
#               in the top-k results
#   MRR@k     — mean reciprocal rank of the first correct chunk
#               within the top-k results (0 if not found)
#   Similarity score — mean cosine similarity of the top-1 result
#   Retrieval latency — mean and std of query time in seconds
#
# All metrics are reported overall and per question category.
# Results saved to evaluation/results/results_retrieval.csv.

import json
import time
import csv
import statistics
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths and config ──────────────────────────────────────────────────────────

ROOT            = Path(__file__).parent.parent
QA_PATH         = ROOT / "data" / "qa_dataset.json"
CHROMA_DB_PATH  = ROOT / "chroma_db"
OUTPUT_PATH     = ROOT / "evaluation" / "results" / "results_retrieval.csv"

EMBEDDING_MODEL = "all-mpnet-base-v2"
COLLECTIONS     = ["supermarket_fixed", "supermarket_semantic"]
K_VALUES        = [1, 3, 5]
K_RETRIEVE      = 5  # always retrieve top-5, then evaluate at k=1,3,5


# ── Relevance check ───────────────────────────────────────────────────────────

def is_relevant(chunk_metadata: dict, chunk_document: str, source_entity: str) -> bool:
    """
    A chunk is relevant if the source_entity appears in its entry_header
    or anywhere in the chunk text.
    - entry_header check: precise match for semantic chunks whose headers
      contain the entity ID (e.g. "ZONE_02 — Fresh Produce Section")
    - text check: catches fixed-size chunks where entity IDs appear in the
      body text but not in the header (which is an arbitrary token boundary)
    SECTION_8 / SECTION_0 are normalised to "SECTION 8" / "SECTION 0".
    """
    entity_normalised = source_entity.replace("_", " ")
    header = chunk_metadata.get("entry_header", "")
    return (source_entity in header
            or entity_normalised in header
            or source_entity in chunk_document)


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(results: list[dict], source_entity: str, k: int) -> float:
    """1.0 if any of the top-k chunks is relevant, else 0.0."""
    for r in results[:k]:
        if is_relevant(r["metadata"], r["document"], source_entity):
            return 1.0
    return 0.0


def mrr_at_k(results: list[dict], source_entity: str, k: int) -> float:
    """Reciprocal rank of the first relevant chunk in top-k, else 0.0."""
    for rank, r in enumerate(results[:k], start=1):
        if is_relevant(r["metadata"], r["document"], source_entity):
            return 1.0 / rank
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    # Load QA dataset
    qa_data   = json.loads(QA_PATH.read_text(encoding="utf-8"))
    questions = qa_data["questions"]
    print(f"Loaded {len(questions)} questions.")

    # Load embedding model
    print(f"Loading embedding model ({EMBEDDING_MODEL}) ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Prepare output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for collection_name in COLLECTIONS:
        method = collection_name.split("_")[1]  # "fixed" or "semantic"
        collection = client.get_collection(collection_name)
        print(f"\nEvaluating {collection_name} ({collection.count()} chunks) ...")

        for q in questions:
            qid      = q["id"]
            question = q["question"]
            entity   = q["source_entity"]
            category = q["category"]

            # Embed query and retrieve top-k — timed
            t0 = time.perf_counter()
            query_embedding = model.encode(question, convert_to_numpy=True).tolist()
            results_raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=K_RETRIEVE,
                include=["metadatas", "distances", "documents"],
            )
            latency = time.perf_counter() - t0

            # Unpack ChromaDB results into a flat list of dicts
            results = [
                {
                    "metadata":  results_raw["metadatas"][0][i],
                    "document":  results_raw["documents"][0][i],
                    "distance":  results_raw["distances"][0][i],
                    "similarity": 1 - results_raw["distances"][0][i],
                }
                for i in range(len(results_raw["ids"][0]))
            ]

            # Compute metrics
            row = {
                "question_id":   qid,
                "category":      category,
                "source_entity": entity,
                "method":        method,
                "latency_s":     round(latency, 6),
                "top1_similarity": round(results[0]["similarity"], 4) if results else 0,
                "top1_chunk_id": results[0]["metadata"]["chunk_id"] if results else "",
            }

            for k in K_VALUES:
                row[f"recall@{k}"]  = recall_at_k(results, entity, k)
                row[f"mrr@{k}"]     = round(mrr_at_k(results, entity, k), 4)

            rows.append(row)

        print(f"  Done.")

    # Write CSV
    fieldnames = [
        "question_id", "category", "source_entity", "method",
        "latency_s", "top1_similarity", "top1_chunk_id",
        "recall@1", "recall@3", "recall@5",
        "mrr@1",    "mrr@3",    "mrr@5",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_PATH}")

    # Print summary
    print_summary(rows)


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    for method in ["fixed", "semantic"]:
        mr = [r for r in rows if r["method"] == method]
        n  = len(mr)
        print(f"\n{method.upper()} — {n} questions")

        for k in K_VALUES:
            recall = sum(r[f"recall@{k}"] for r in mr) / n
            mrr    = sum(r[f"mrr@{k}"]    for r in mr) / n
            print(f"  Recall@{k}: {recall:.4f}   MRR@{k}: {mrr:.4f}")

        lats = [r["latency_s"] for r in mr]
        print(f"  Top-1 similarity: {sum(r['top1_similarity'] for r in mr)/n:.4f}")
        print(f"  Latency: {statistics.mean(lats):.4f}s mean / {statistics.stdev(lats):.4f}s std")

        print(f"  Per-category Recall@5:")
        for cat in sorted(set(r["category"] for r in mr)):
            cat_rows = [r for r in mr if r["category"] == cat]
            print(f"    {cat:<30} {sum(r['recall@5'] for r in cat_rows)/len(cat_rows):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    run()