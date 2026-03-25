from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from retrieval import retrieve
from load import load_embeddings
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = OpenAI(api_key=api_key)


class QuestionRequest(BaseModel):
    question: str

def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

summary = load_text("summary.txt")

@app.post("/ask")
def ask_question(req: QuestionRequest):
    user_question = req.question

    results = retrieve(
        query=user_question,
        chunks=load_embeddings(),
        model=model,
        k=3)

    context = "\n\n".join([ r["text"] for r in results])

    def stream():
        with client.responses.stream(
            model="gpt-4o-mini",
            input=f"""
                Context:
                {context,summary}

                Question:
                {user_question}
                """
        )as response:
            for event in response:
                if event.type=="response.output_text.delta":
                    yield event.delta

    return StreamingResponse(stream(), media_type="text/plain")


