from retrieval import retrieve
from load import load_embeddings
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from intent_detector import route_query
import os
from openai import OpenAI
from intent_types import answer_single
import json
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_question = "Summarize Episodes Akak the testing war and client"


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks =[]

with open("./data/chunks/chunks.jsonl","r",encoding="utf-8") as f:
    for line in f:
        if line.split():
            chunks.append(json.loads(line))

# user_question = "Risks shaping the plastic resin market in 2026 with RTi Global"

# results = retrieve(
#     query=user_question,
#     chunks=load_embeddings(),
#     model=model) # k-total chunks and n-top chunks are hard coded to 20 and 5 respectively

# # print(results)
# context = "\n\n".join([ f'{r["text"]}  {r["day"]}' for r in results])

# print(context)


response = {'route': 'single', 'speakers': [], 'dates': []}

def answer(user_question, chunks, model,client) -> str:
    route = route_query(user_question, client)
    intent = route["route"]
    speakers = route.get("speakers", [])

    if intent == "analytical":
        pass

    elif intent == "temporal":
        # naive extraction — improve with NER or regex for your date formats
       pass

    elif intent == "multi":
        pass

    elif intent == "single":
        return answer_single(user_question, route, chunks, model)
    

print(answer(user_question, chunks,model,client))




