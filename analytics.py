# analytics.py
from retrieval import filter_chunks

def analyze_speaker(chunks, speaker_name: str, client, question: str) -> str:
    speaker_chunks = filter_chunks(chunks, speakers=[speaker_name])
    if not speaker_chunks:
        return f"No chunks found for speaker: {speaker_name}"

    all_text = "\n\n".join(
        f"[{c.get('title')} | {c.get('day')}]\n{c['text']}"
        for c in speaker_chunks[:30]   # cap tokens
    )

    word_counts = {c.get("title", "?"): c["token_count"] for c in speaker_chunks}

    analysis_prompt = f"""
You are analyzing podcast transcripts. Speaker: {speaker_name}

Transcript excerpts (sorted chronologically):
{all_text}

Answer this analytical question: {question}

If sentiment is asked: rate as positive/neutral/negative with evidence.
If comparing speakers: list concrete differences in topic focus and tone.
If word count or frequency is asked: use the chunk data provided.
Be concise and cite episode titles.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    return resp.choices[0].message.content