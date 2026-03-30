import json
# query_router.py
ROUTE_PROMPT = """
Classify this question into exactly one category:

- single: asks about one specific episode, speaker, or fact
- multi: asks across all episodes or about patterns over time
- temporal: compares two time periods or asks "what changed"
- analytical: asks for stats, sentiment, or speaker comparisons

Question: {question}

Return JSON: {{"route": "single|multi|temporal|analytical", "speakers": ["..."], "dates": ["..."]}}
"""

def route_query(question: str, client) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ROUTE_PROMPT.format(question=question)}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

