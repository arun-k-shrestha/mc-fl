import json

def answer_single(question, route, data, client):
    speakers = route.get("speakers", [])
    dates = route.get("dates", [])

    instructions = """
    You answer questions using only the provided context.

    Rules:
    - Use only the provided data.
    - Prioritize content relevant to the listed speakers and dates.
    - If the data is sufficient, return:
    {"response": true, "text": "..."}
    - If the data is insufficient, ambiguous, or missing key facts, return:
    {"response": false, "questions": ["...", "..."]}
    - Do not invent facts.
    - Keep answers concise and grounded.
    - Each follow-up question must target one missing fact and be retrieval-friendly.

    """

    user_input = {
        "question": question,
        "data": data,
        "speakers": speakers,
        "dates": dates,
    }

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=json.dumps(user_input, ensure_ascii=False)
    )

    return json.loads(response.output_text)


def answer_multi(question, route, data, client):
    speakers = route.get("speakers", [])
    dates = route.get("dates", [])

    instructions = """
    You answer questions using only the provided context. The user may ask about patterns across all episodes or trends over time. Your goal is to determine whether the available data is sufficient to answer, or if more targeted retrieval is needed.

    Rules:

    Use only the provided data.
    Do not invent or assume missing facts.
    Keep answers concise and grounded in the data.

    Decision:

    If the data is sufficient to clearly answer (e.g., title, summary, or direct factual lookup), return:
    {"response": true, "text": "..."}
    If the data is insufficient, ambiguous, missing key details, or requires broader coverage (e.g., trends across episodes), return:
    {"response": false, "questions": ["...", "..."]}

    Follow-up Question Requirements:

    Generate multiple questions when the request involves distinct aspects or broader analysis.
    Each question MUST include: speaker name, title, and date (only if available in the data — do not fabricate).
    Each question should be retrieval-friendly and reuse wording from the provided data (aligned with embeddings).
    Avoid vague or generic phrasing.

    Additional Notes:

    Prefer precise, data-aligned queries over broad ones.
    Ensure all outputs are valid JSON only (no extra text outside the JSON).
    """

    user_input = {
        "question": question,
        "data": data,
        "speakers": speakers,
        "dates": dates,
    }

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=json.dumps(user_input, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_multi_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "boolean"
                        },
                        "text": {
                            "type": "string"
                        },
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["response", "text", "questions"],
                    "additionalProperties": False
                }
            }
        }
    )

    return json.loads(response.output_text)


def answer_temporal(question, route, data, client):
    speakers = route.get("speakers", [])
    dates = route.get("dates", [])


    instructions = """
The user asks a temporal question (e.g., comparing two time periods or asking “what changed”).

Your goal is to determine whether the available data is sufficient to perform a valid comparison, or if additional retrieval is required.

Rules:
- Use only the provided data.
- Do not invent or assume missing facts.
- Keep answers concise and grounded in the data.

Decision Criteria:
Data is sufficient ONLY if:
- Both time periods (or points of comparison) are explicitly present
- Comparable attributes (e.g., topic, speaker, metrics) exist across them
- All comparison points are present
- The level of detail in the data matches the level requested by the user
- The answer can be supported with multiple specific, concrete details from the data

If the data is sufficient, return:
{"response": true, "text": "..."}

If the data is insufficient (missing one period, missing comparable attributes, ambiguous, or requiring broader coverage), return:
{"response": false, "questions": ["...", "..."]}

Follow-up Question Requirements:
- This is a temporal comparison → explicitly separate what is being compared
- Each question MUST include: speaker name, title, and date (only if available in the data — do not fabricate).
- Each question must be retrieval-friendly and reuse title from the provided data
- Include speaker name, title, and date ONLY if explicitly present (do not fabricate)
- Avoid vague or generic phrasing

Additional Notes:
Prefer precise, data-aligned queries over broad ones.
Ensure all outputs are valid JSON only (no extra text outside the JSON).
"""

    user_input = {
        "question": question,
        "data": data,
        "speakers": speakers,
        "dates": dates,
    }

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=json.dumps(user_input, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_multi_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "boolean"
                        },
                        "text": {
                            "type": "string"
                        },
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["response", "text", "questions"],
                    "additionalProperties": False
                }
            }
        }
    )

    return json.loads(response.output_text)


# asks for stats, sentiment, or speaker comparisons
def answer_analytical(question, route, data, client):
    speakers = route.get("speakers", [])
    dates = route.get("dates", [])

    instructions = """
    You answer analytics-focused questions using only the provided context. The user may ask for statistics, speaker comparisons, or sentiment analysis.

    You must be conservative. Default to insufficient data unless the required information is explicitly present.

    Rules:

    * Use only the provided data.
    * Do not infer, estimate, or assume missing information.
    * Do not generalize from partial coverage.
    * Keep answers concise and strictly grounded in the data.

    Critical Constraint:

    No sentiment data is available in the provided context.
    Therefore, any request that asks for sentiment, tone, opinion polarity, emotional valence, or sentiment comparison MUST return:
    {"response": false, "questions": ["..."]}

    Decision Criteria:

    Return {"response": true, "text": "..."} ONLY if ALL of the following are true:

    The request is limited to factual analytics supported by the provided data.
    All required statistics are explicitly present or directly computable from the data.
    All speakers referenced in the question are present in the data.
    Comparable attributes exist across all entities being compared.
    The provided data fully covers the requested scope.

    Return {"response": false, "questions": ["...", "..."]} if ANY of the following are true:

    The request involves sentiment in any form.
    A required metric is missing.
    A speaker, topic, date, or comparison dimension is missing.
    The request requires broader coverage than the provided data.
    The request requires trends, aggregation, or comparisons across items not fully represented in the context.
    There is any ambiguity about whether the data is complete enough.

    Bias Rule:

    When uncertain, return false.
    Prefer false over a partial or weakly supported answer.

    Follow-up Question Requirements:

    Generate multiple questions when the request has multiple missing dimensions.
    - Each question MUST include: speaker name, title, and date (only if available in the data — do not fabricate).
    - Each question must be retrieval-friendly and reuse title from the provided data
    - Include speaker name, title, and date only when available in the context.

    Additional Notes:
    Prefer precise, data-aligned queries over broad ones.
    Ensure all outputs are valid JSON only (no extra text outside the JSON).
  """

    user_input = {
        "question": question,
        "data": data,
        "speakers": speakers,
        "dates": dates,
    }

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=json.dumps(user_input, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_multi_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "boolean"
                        },
                        "text": {
                            "type": "string"
                        },
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["response", "text", "questions"],
                    "additionalProperties": False
                }
            }
        }
    )

    return json.loads(response.output_text)


# candidates = chunks
# speakers = route.get("speakers", [])
# dates = route.get("dates", [])
# keywords = route.get("keywords", [])
# episode_hints = route.get("episode_hints", [])

# if speakers:
#     candidates = [
#         c for c in candidates
#         if any(s.lower() in [sp.lower() for sp in c.get("speakers", [])] for s in speakers)
#     ]

# if dates:
#     candidates = [
#         c for c in candidates
#         if any(d.lower() in (c.get("day") or "").lower() for d in dates)
#     ] or candidates

# if episode_hints:
#     narrowed = [
#         c for c in candidates
#         if any(h.lower() in (c.get("title") or "").lower() for h in episode_hints)
#     ]
#     if narrowed:
#         candidates = narrowed

# retrieval_query = " ".join(speakers + episode_hints + keywords + [question]).strip()

# results = retrieve(
#     query=retrieval_query,
#     chunks=candidates,
#     model=model,
#     k=8,
#     top_n=4
# )
