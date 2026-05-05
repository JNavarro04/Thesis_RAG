# Collections created:
#   supermarket_fixed    — fixed-size chunks  (chunks_fixed.json)
#   supermarket_semantic — semantic chunks    (chunks_semantic.json)
#
# Indexing latency is saved to evaluation/indexing_latency.json

import json
import time
from pathlib import Path
 
import chromadb
from sentence_transformers import SentenceTransformer
 
ROOT            = Path(__file__).parent.parent
EMBEDDING_MODEL = "all-mpnet-base-v2"
CHROMA_DB_PATH  = ROOT / "chroma_db"
LATENCY_PATH    = ROOT / "evaluation" / "indexing_latency.json"
 
COLLECTIONS = {
    "supermarket_fixed":    ROOT / "data" / "chunks_fixed.json",
    "supermarket_semantic": ROOT / "data" / "chunks_semantic.json"}
 
 
def run():
    #Load embedding model once, shared across both collections
    print(f"Loading embedding model ({EMBEDDING_MODEL})")
    model = SentenceTransformer(EMBEDDING_MODEL)
 
    #Initialise ChromaDB
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
 
    latency = {}
 
    for collection_name, chunks_path in COLLECTIONS.items():
        method = collection_name.split("_")[1]  #fixed or semantic
 
        #Load chunks
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        print(f"\n{collection_name}: {len(chunks)} chunks")
 
        #Embed
        print("Embedding chunks...")
        t0 = time.perf_counter()
        embeddings = model.encode(
            [c["text"] for c in chunks],
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32,
        ).tolist()
        embed_time = time.perf_counter() - t0
        print(f"Embedding time: {embed_time:.2f}s")
 
        #Recreate collection from scratch 
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name) #delete current

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}) #create new colelction
 
        #Insert
        t0 = time.perf_counter()
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[{
                "chunk_id":        c["chunk_id"],
                "token_count":     c["token_count"],
                "entry_header":    c.get("entry_header", ""),
                "chunking_method": method,
            } for c in chunks],
        )
        index_time = time.perf_counter() - t0
        print(f"  Indexing time : {index_time:.2f}s")
        print(f"  Stored        : {collection.count()} documents")
 
        latency[f"{method}_embedding_seconds"] = round(embed_time, 4)
        latency[f"{method}_indexing_seconds"]  = round(index_time, 4)
        latency[f"{method}_total_seconds"]     = round(embed_time + index_time, 4)
 
    #Save latency to file
    LATENCY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATENCY_PATH.write_text(json.dumps(latency, indent=2))
 
    print("\nIndexing latency summary:")
    for method in ["fixed", "semantic"]:
        print(f"  {method:8s} — embed: {latency[f'{method}_embedding_seconds']}s"
              f"  |  index: {latency[f'{method}_indexing_seconds']}s"
              f"  |  total: {latency[f'{method}_total_seconds']}s")
    print(f"\nSaved latency to {LATENCY_PATH}")
    print("Indexing complete. ChromaDB collections are ready.")
 
 
if __name__ == "__main__":
    run()