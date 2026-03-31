# main.py (refactored)
from intent_detector import route_query
from retrieval import retrieve_filtered
from load import load_embeddings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os, json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = load_embeddings()

def answer(user_question: str) -> str:
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

    else:  # single
        pass


print(answer("who is Nicole?"))