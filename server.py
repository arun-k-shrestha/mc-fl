from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json
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

prompt_data = ""
with open("prompt_data.txt", "r") as f:
    prompt_data = f.read()



api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = load_embeddings()  # load once

client = OpenAI(api_key=api_key)

class QuestionRequest(BaseModel):
    question: str

def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
def ask_question(req: QuestionRequest):
    user_question = req.question
    prompt_1 = f"""
            You generate search queries for an embeddings database.

            User question:
            {user_question}

            Dataset schema / description:
            {prompt_data}

            Instructions:
            - Generate 1 to 3 concise search queries.
            - Only generate more than 3 if the question clearly requires multiple distinct aspects.
            - Each query MUST include (speaker name, title, date)
            - Keep queries specific and retrieval-focused.
            - If no valid query can be formed using the dataset fields, return an empty JSON array [].

            Output:
            Return a JSON array of strings only. Do not include any explanation.
            """

    response_1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_1}],
        response_format={"type": "json_object"}
    )

    content = response_1.choices[0].message.content
    data = json.loads(content)
    questions = data.get("queries", [])

    context = []
    for question in questions:
        results = retrieve(
            query=question,
            chunks=embeddings,
            model=model)
        context.append(results)

    context = "\n\n".join([ r["text"] for r in results])

    def stream():
        with client.responses.stream(
            model="gpt-4o-mini",
            instructions="""
                    You answer questions about McKinney Flavelle's Hot Commodity Podcast using only the provided context.

                    Behavior:
                    - Answer only from the context.
                    - Do not rely on outside knowledge.
                    - Do not guess.
                    - If the answer is not in the context, say: "I couldn't find that in the provided podcast context."
                    - If the context is incomplete, say what part is missing.
                    - Mention speaker names when the context supports them.
                    - Mention episode title or date when clearly available in the context.
                    - Prefer exact factual answers over broad summaries.
                    - Keep the response clear and concise.
                    """,
            input=f"""
                Context:
                {context}

                Question:
                {user_question}
                """
        )as response:
            for event in response:
                if event.type=="response.output_text.delta":
                    yield event.delta

    return StreamingResponse(stream(), media_type="text/plain")


