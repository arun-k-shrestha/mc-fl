# main.py (refactored)
from query_router import route_query
from aggregation import retrieve_all_episodes
from temporal import retrieve_temporal_comparison
from analytics import analyze_speaker
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
        return analyze_speaker(embeddings, speakers[0] if speakers else "", client, user_question)

    elif intent == "temporal":
        # naive extraction — improve with NER or regex for your date formats
        dates = route.get("dates", [])
        period_a = dates[0] if len(dates) > 0 else ""
        period_b = dates[1] if len(dates) > 1 else ""
        periods = retrieve_temporal_comparison(user_question, embeddings, model, period_a, period_b, speakers or None)
        context = "=== Period A ===\n" + "\n\n".join(c["text"] for c in periods["period_a"])
        context += "\n\n=== Period B ===\n" + "\n\n".join(c["text"] for c in periods["period_b"])

    elif intent == "multi":
        results = retrieve_all_episodes(user_question, embeddings, model, speakers=speakers or None)
        context = "\n\n".join(
            f"[{c.get('title')} | {c.get('day')}]\n{c['text']}" for c in results
        )

    else:  # single
        results = retrieve_filtered(user_question, embeddings, model, speakers=speakers or None)
        context = "\n\n".join(c["text"] for c in results)

    # Final LLM call (unchanged from your original)
    resp = client.responses.create(
        model="gpt-4o-mini",
        instructions="Answer using ONLY the provided context from McKinney Flavelle's Hot Commodity Podcast.",
        input=f"Context:\n{context}\n\nQuestion: {user_question}"
    )
    return resp.output_text

print(answer("who is Nicole?"))