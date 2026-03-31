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

user_question = ""


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks =[]

with open("./data/chunks/chunks.jsonl","r",encoding="utf-8") as f:
    for line in f:
        if line.split():
            chunks.append(json.loads(line))





