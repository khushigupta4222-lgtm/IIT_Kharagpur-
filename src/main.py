print(">>> main.py file loaded")

import sys
import json
import numpy as np
from pathlib import Path

# ---------------- PROJECT ROOT ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ---------------- IMPORTS ----------------
from src.ingestion.load_data import load_example
from src.chunking.chunker import chunk_text
from src.retrieval.hybrid import hybrid_retrieval


def main():
    print(">>> MAIN.PY IS RUNNING <<<")

    # ---------- INGESTION ----------
    example_dir = PROJECT_ROOT / "data" / "Example_001"
    novel, backstory = load_example(str(example_dir))

    print("Novel loaded:", len(novel))
    print("Backstory loaded:", len(backstory))

    # ---------- CHUNKING ----------
    chunks = chunk_text(novel)
    print("Chunks created:", len(chunks))

    # ---------- LOAD CHUNKS ----------
    chunks_path = PROJECT_ROOT / "data" / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # ---------- LOAD EMBEDDINGS ----------
    embeddings_path = PROJECT_ROOT / "data" / "chunk_embeddings.npy"
    chunk_embeddings = np.load(embeddings_path)
    print("Chunk embeddings shape:", chunk_embeddings.shape)

    # ---------- LOAD QUERY EMBEDDING ----------
    query_embedding_path = PROJECT_ROOT / "data" / "query_embedding.npy"
    query_embedding = np.load(query_embedding_path)
    print("Query embedding shape:", query_embedding.shape)

    # ---------- QUERY ----------
    query = "Why did the protagonist betray his friend?"

    # ---------- HYBRID RETRIEVAL ----------
    results = hybrid_retrieval(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        query_embedding=query_embedding,
        query=query,
        top_k=3,
        alpha=0.65
    )

    # ---------- TEMP DEBUG OUTPUT ----------
    print("\nTop retrieved chunks:\n")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"Rank {i} | Score: {score:.4f}")
        print(chunk[:500])
        print("-" * 80)

    # ---------- BASELINE PLACEHOLDER ----------
    prediction = 1
    print("\nBaseline prediction:", prediction)


if __name__ == "__main__":
    main()
