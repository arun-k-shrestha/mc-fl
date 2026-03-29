from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, chunks, top_n=5):
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker_model.predict(pairs)

    scored = list(zip(scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored[:top_n]]