import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b)

def retrieve(query, chunks, model, k=5):
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for chunk in chunks:
        emb = np.array(chunk["embedding"])
        score = cosine_sim(q_emb, emb)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]