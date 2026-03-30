import numpy as np
from reranker import rerank

def cosine_sim(a, b):
    return float(np.dot(a, b))

def retrieve(query, chunks, model, k=10,top_n=4, duplicate_threshold =0.92):
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for chunk in chunks:
        emb = np.array(chunk["embedding"])
        score = cosine_sim(q_emb, emb)
        scored.append((score, chunk,emb))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, chunk, emb in scored:
        duplicate = False
        for _, _, prev_emb in selected:
            if cosine_sim(emb, prev_emb) > duplicate_threshold:
                duplicate = True
                break

        if not duplicate:
            selected.append((score, chunk, emb))

        if len(selected) == k:
            break
    top_chunks = [chunk for _, chunk, _ in selected]
    return rerank(query,top_chunks,top_n=top_n)