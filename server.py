from fastapi import FastAPI, Header
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from typing import Any, Dict, List, Optional

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
chunks = load_embeddings()

client = OpenAI(api_key=api_key)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(req: QuestionRequest):
    user_question = req.question

    results = retrieve(
        query=user_question,
        chunks=chunks,
        model=model,
        k=3)

    context = "\n\n".join([ r["text"] for r in results])

    def stream():
        with client.responses.stream(
            model="gpt-4o-mini",
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


# Vapi

class ToolCallFunction(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}

class ToolCall(BaseModel):
    id: str
    function: ToolCallFunction

class VapiMessage(BaseModel):
    type: str
    toolCallList: Optional[List[ToolCall]] = None

class VapiPayload(BaseModel):
    message: VapiMessage


def answer_from_docs(user_question: str) -> str:
    results = retrieve(
        query=user_question,
        chunks=chunks,
        model=model,
        k=3,
    )

    if not results:
        return "I do not have enough information."

    context = "\n\n".join([r["text"] for r in results if "text" in r and r["text"]])

    if not context.strip():
        return "I do not have enough information."

    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
You answer questions using only the provided context.
If the context is insufficient, say exactly: "I do not have enough information."

Context:
{context}

Question:
{user_question}
""".strip(),
    )

    text = getattr(response, "output_text", None)
    if text and text.strip():
        return text.strip()

    return "I do not have enough information."


@app.post("/vapi")
def vapi_tool_handler(
    payload: VapiPayload,
    x_vapi_secret: Optional[str] = Header(default=None),
):
    message = payload.message

    if message.type != "tool-calls":
        return {"results": []}

    if not message.toolCallList:
        return {"results": []}

    results = []

    for tool_call in message.toolCallList:
        tool_name = tool_call.function.name
        args = tool_call.function.arguments or {}

        if tool_name != "ask_docs":
            results.append(
                {
                    "toolCallId": tool_call.id,
                    "result": "Unsupported tool.",
                }
            )
            continue

        question = str(args.get("question", "")).strip()

        if not question:
            answer = "I do not have enough information."
        else:
            try:
                answer = answer_from_docs(question)
            except Exception as e:
                results.append(
                    {
                        "toolCallId": tool_call.id,
                        "error": f"Tool failed: {str(e)}",
                    }
                )
                continue

        results.append(
            {
                "toolCallId": tool_call.id,
                "result": answer.replace("\n", " ").strip(),
            }
        )

    return {"results": results}