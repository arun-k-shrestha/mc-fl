import json

def load_embeddings():
    chunks = []
    with open("chunks_with_embeddings.jsonl", "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

