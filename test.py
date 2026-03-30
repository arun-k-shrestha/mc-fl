from retrieval import retrieve
from load import load_embeddings
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from query_router import route_query
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(route_query("what did Nicole disscued?",client))

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# user_question = "Risks shaping the plastic resin market in 2026 with RTi Global"

# results = retrieve(
#     query=user_question,
#     chunks=load_embeddings(),
#     model=model) # k-total chunks and n-top chunks are hard coded to 20 and 5 respectively

# # print(results)
# context = "\n\n".join([ f'{r["text"]}  {r["day"]}' for r in results])

# print(context)





