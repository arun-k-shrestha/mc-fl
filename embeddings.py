import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks =[]

with open("./data/chunks/chunks.jsonl","r",encoding="utf-8") as f:
    for line in f:
        if line.split():
            chunks.append(json.loads(line))

def build_input(c):
    parts = [
        f"Title: {c['title']}" if c.get("title") else "",
        f"Summary: {c['summary']}" if c.get("summary") else "",
        f"Content: {c['text']}"
    ]
    return "\n".join(p for p in parts if p)

texts = [build_input(c) for c in chunks]
embeddings = model.encode(texts,normalize_embeddings=True)

for chunk,emb in zip(chunks,embeddings):
    chunk["embedding"] = emb.tolist()

with open("chunks_with_embeddings.jsonl", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk,ensure_ascii=False)+ "\n")

