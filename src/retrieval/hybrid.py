from src.retrieval.tfidf import tfidf_retrieval
from src.retrieval.embeddings import embedding_retrieval


def normalize(score_dict):
    """
    Min-max normalize scores to [0, 1]
    """
    if not score_dict:
        return score_dict

    values = list(score_dict.values())
    min_v, max_v = min(values), max(values)

    if max_v - min_v == 0:
        return score_dict

    return {
        k: (v - min_v) / (max_v - min_v)
        for k, v in score_dict.items()
    }


def hybrid_retrieval(
    chunks,
    chunk_embeddings,
    query_embedding,
    query,
    top_k=3,
    alpha=0.65,
    adjacency_boost=0.05,   # Step 5
):
    """
    alpha        -> weight for embeddings
    (1 - alpha)  -> weight for TF-IDF
    """

    # ---------- 1. Wide candidate retrieval ----------
    tfidf_results = tfidf_retrieval(
        chunks,
        query,
        top_k=top_k * 3
    )

    embed_results = embedding_retrieval(
        chunks,
        chunk_embeddings,
        query_embedding,
        top_k=top_k * 3
    )

    # ---------- 2. Separate score maps ----------
    tfidf_scores = {chunk: score for chunk, score in tfidf_results}
    embed_scores = {chunk: score for chunk, score in embed_results}

    # ---------- 3. Normalize ----------
    tfidf_scores = normalize(tfidf_scores)
    embed_scores = normalize(embed_scores)

    # ---------- 4. Fuse ----------
    fused_scores = {}
    for chunk in set(tfidf_scores) | set(embed_scores):
        fused_scores[chunk] = (
            (1 - alpha) * tfidf_scores.get(chunk, 0.0)
            + alpha * embed_scores.get(chunk, 0.0)
        )

    # ---------- 5. Adjacency boost (narrative continuity) ----------
    chunk_index = {chunk: i for i, chunk in enumerate(chunks)}

    for chunk in list(fused_scores.keys()):
        idx = chunk_index.get(chunk)
        if idx is None:
            continue

        for neighbor in (idx - 1, idx + 1):
            if 0 <= neighbor < len(chunks):
                neighbor_chunk = chunks[neighbor]
                if neighbor_chunk in fused_scores:
                    fused_scores[neighbor_chunk] += adjacency_boost

    # ---------- 6. Final ranking ----------
    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]
