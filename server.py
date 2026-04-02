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
from intent_router import answer


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
    response = answer(user_question,prompt_data,client,model)
    print(response)

    boolean = response.get("response", False)
    questions = response.get("questions", [])
    text = response.get("text", "")

    if boolean:
        context = text
    else:
        context = []
        for question in questions:
            results = retrieve(
                query=question,
                chunks=embeddings,
                model=model)
            context.extend(results)

        context = "\n\n".join([ r["text"] for r in context])

    def stream():
        with client.responses.stream(
            model="gpt-4o-mini",
            instructions="""You answer questions about McKinney Flavelle's Hot Commodity Podcast using ONLY the provided transcript context.
                    SOURCE RULES:
                    - The transcript in the Context section is the ONLY source of truth.
                    - The Context may contain either:
                        1. verbatim transcript excerpts, or
                        2. directly extracted answer text from an earlier grounded step.
                    - Do NOT use outside knowledge.

                    DECISION LOGIC:
                    - If the answer is explicitly supported in the transcript, provide a detailed answer.

                    ANSWER REQUIREMENTS:
                    - Write a natural, well-explained response in paragraph form.
                    - Use multiple sentences (at least 4 when enough information is available).
                    - Expand the answer by incorporating all relevant details from the transcript, not just the first matching line.

                    ATTRIBUTION:
                    - Include speaker name(s) only if explicitly mentioned in the transcript.
                    - Include episode title or date only if explicitly supported by the transcript.

                    CONSTRAINTS:
                    - Do NOT add reasoning, interpretation, assumptions, or outside knowledge.
                    - If additional relevant details exist in the transcript, include them to make the answer more complete.

                    STYLE:
                    - Write like a clear, helpful ChatGPT response.
                    - Use smooth, connected sentences (not bullet points or sections).
                    - Be specific and informative, but not verbose or repetitive.
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
                    yield f"data: {event.delta}\n\n"
            yield "event: done\ndata: [DONE]\n\n"    

    return StreamingResponse(
    stream(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    },
    )


