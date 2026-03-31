import numpy as np
from reranker import rerank


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def retrieve(query, chunks, model, k=10, top_n=4, duplicate_threshold=0.92):
    if not chunks:
        return []

    q_emb = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for chunk in chunks:
        emb = np.asarray(chunk["embedding"], dtype=np.float32)
        score = cosine_sim(q_emb, emb)
        scored.append((score, chunk, emb))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for score, chunk, emb in scored:
        is_duplicate = False

        for _, _, prev_emb in selected:
            if cosine_sim(emb, prev_emb) > duplicate_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append((score, chunk, emb))

        if len(selected) >= k:
            break

    top_chunks = [chunk for _, chunk, _ in selected]
    return rerank(query, top_chunks, top_n=top_n)