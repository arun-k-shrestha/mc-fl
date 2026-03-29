import numpy as np
from reranker import rerank

def cosine_sim(a, b):
    return np.dot(a, b)

def retrieve(query, chunks, model, k=2,top_n=1):
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for chunk in chunks:
        emb = np.array(chunk["embedding"])
        score = cosine_sim(q_emb, emb)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:k]]
    return rerank(query,top_chunks,top_n=top_n)