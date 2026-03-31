from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import os
from retrieval import retrieve
from load import load_embeddings
from sentence_transformers import SentenceTransformer
from intent_router import answer



load_dotenv()

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



user_question = "what are the things Nicole spoke?"

answer_1 = answer(user_question,prompt_data,client,model)

print(answer_1) 

boolean = answer_1.get("response", False)
questions = answer_1.get("questions", [])
text = answer_1.get("text", "")

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

response_2 = client.responses.create(
    model="gpt-4o-mini",
    instructions="""You answer questions about McKinney Flavelle's Hot Commodity Podcast using ONLY the provided transcript context.
        SOURCE RULES:
        - The transcript in the Context section is the ONLY source of truth.
        - The "Relevant Speaker, Title, and Date" section is for attribution only and must NOT be used to introduce new facts.
        - Do NOT use outside knowledge.
        - Do NOT guess, infer, or fill in missing information.

        DECISION LOGIC:
        - If the answer is explicitly supported in the transcript, provide a detailed answer.
        - If the answer is NOT present at all, respond exactly:
        I couldn't find that in the provided podcast context.
        - If the topic is mentioned but lacks enough detail, respond:
        The provided transcript is incomplete for this question. Missing information: <brief description>

        ANSWER REQUIREMENTS:
        - Write a natural, well-explained response in paragraph form.
        - Use multiple sentences (at least 4 when enough information is available).
        - Expand the answer by incorporating all relevant details from the transcript, not just the first matching line.
        - Stay strictly grounded in what is explicitly stated in the transcript.

        ATTRIBUTION:
        - Include speaker name(s) only if explicitly mentioned in the transcript.
        - Include episode title or date only if explicitly supported by the transcript.

        CONSTRAINTS:
        - Do NOT add reasoning, interpretation, assumptions, or outside knowledge.
        - Do NOT generalize beyond what is stated.
        - If additional relevant details exist in the transcript, include them to make the answer more complete.

        STYLE:
        - Write like a clear, helpful ChatGPT response.
        - Use smooth, connected sentences (not bullet points or sections).
        - Be specific and informative, but not verbose or repetitive.
        """,
        input=f"""
    Context:
    {context}

    Relevant Speaker, Title, and Date:
    {questions}

    Question:
    {user_question}
    """
    )

print(response_2.output_text)


