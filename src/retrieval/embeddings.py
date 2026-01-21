import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def embedding_retrieval(chunks, chunk_embeddings, query_embedding, top_k=5):
    """
    Uses precomputed embeddings only.
    No torch, no transformers.
    """
    scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        chunk_embeddings
    )[0]

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]
