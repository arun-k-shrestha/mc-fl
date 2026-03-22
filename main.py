from openai import OpenAI
from dotenv import load_dotenv
import os
from retrieval import retrieve
from load import load_embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = OpenAI(api_key=api_key)


user_question = "What emotions does the speaker express?"

results = retrieve(
    query=user_question,
    chunks=load_embeddings(),
    model=model,
    k=3)

context = "\n\n".join([ r["text"] for r in results])


response = client.responses.create(
    model="gpt-4o-mini",
    input=f"""
        Context:
        {context}

        Question:
        {user_question}
        """
)


print(response.output_text)
