from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import os
from retrieval import retrieve
from load import load_embeddings
from sentence_transformers import SentenceTransformer


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



user_question = " which episodes talked about the war"
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
    - - If no valid query can be formed using the dataset fields, return: {{"queries": []}}

    Output:
    Return JSON in exactly this format:
    {{"queries": ["query 1", "query 2"]}}   Do not include any explanation.
    """

response_1 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": prompt_1}],
response_format={"type": "json_object"}
)

content = response_1.choices[0].message.content
data = json.loads(content)
questions = data.get("queries", [])

print(questions)

context = []
for question in questions:
    results = retrieve(
        query=question,
        chunks=embeddings,
        model=model)
    context.extend(results)

context = "\n\n".join([ r["text"] for r in context])

print(context)

response_2 = client.responses.create(
    model="gpt-4o-mini",
    instructions="""
    You answer questions about McKinney Flavelle's Hot Commodity Podcast using ONLY the provided context.

    Rules:
    - Use ONLY the transcript content in the Context section as the source of truth.
    - The "Relevant Speaker, Title, and Date" section is for attribution only, not for generating new facts.
    - Do NOT use outside knowledge.
    - Do NOT guess or infer beyond what is explicitly stated.

    If the answer IS found:
    - Provide a clear, concise, factual answer.
    - Include speaker name(s) if mentioned in the context.
    - Include episode title or date if clearly supported.

    If the answer is NOT found:
    - Respond exactly with:
    "I couldn't find that in the provided podcast context."

    If the context is incomplete:
    - State what specific information is missing (e.g., "The transcript does not include discussion of X").

    Style:
    - Be precise and concise.
    - Prefer direct answers over summaries.
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


