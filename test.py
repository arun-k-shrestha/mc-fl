from retrieval import retrieve
from load import load_embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

user_question = "summarize Latest updates on weather, war, wheat, economy, & inflation"

results = retrieve(
    query=user_question,
    chunks=load_embeddings(),
    model=model) # k-total chunks and n-top chunks are hard coded to 20 and 5 respectively

context = "\n\n".join([ r["text"] for r in results])

print(context)




