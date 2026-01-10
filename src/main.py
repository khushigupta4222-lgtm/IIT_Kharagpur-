print(">>> main.py file loaded")

import sys
from pathlib import Path

# ---------------------------
# Project root setup
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ---------------------------
# Imports
# ---------------------------
from src.ingestion.load_data import load_example
from src.chunking.chunker import chunk_text

from src.retrieval.tfidf import tfidf_retrieval
from src.retrieval.hybrid import hybrid_retrieval

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    print(">>> MAIN.PY IS RUNNING <<<")

    # ---- Ingestion ----
    example_dir = PROJECT_ROOT / "data" / "example_001"
    novel, backstory = load_example(str(example_dir))

    print("Novel loaded successfully.")
    print("Novel length:", len(novel))

    print("\nBackstory loaded successfully.")
    print("Backstory length:", len(backstory))

    # ---- Chunking ----
    chunks = chunk_text(novel)

    print("\nChunking done.")
    print("Total chunks created:", len(chunks))

    # ---- LEVEL 3: Hybrid Retrieval (TF-IDF only for now) ----
    query = "Why did the protagonist betray his friend?"

    top_chunks = hybrid_retrieval(
        chunks=chunks,
        query=query,
        tfidf_retrieval_fn=tfidf_retrieval,
        embedding_retrieval_fn=None,   # embeddings OFF for now
        embedder=None,
        alpha=1.0,                     # pure TF-IDF (controlled)
        top_k=3
    )

    print("\nTop retrieved chunks:\n")
    for i, (chunk, score) in enumerate(top_chunks, start=1):
        print(f"Rank {i} | Score: {score:.4f}")
        print(chunk[:500])
        print("-" * 80)

    # ---- Baseline prediction (kept intentionally) ----
    prediction = 1
    print("\nBaseline prediction:", prediction)


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()

