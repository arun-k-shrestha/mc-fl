# temporal.py
from retrieval import filter_chunks,retrieve

def retrieve_temporal_comparison(query, chunks, model, period_a: str, period_b: str, speakers=None):
    """
    period_a / period_b: month prefixes like "Jan 2026", "Mar 2026"
    Returns {"period_a": [...chunks], "period_b": [...chunks]}
    """
    def match_period(c, period):
        return period.lower() in (c.get("day") or "").lower()

    a_chunks = [c for c in chunks if match_period(c, period_a)]
    b_chunks = [c for c in chunks if match_period(c, period_b)]
    if speakers:
        a_chunks = filter_chunks(a_chunks, speakers=speakers)
        b_chunks = filter_chunks(b_chunks, speakers=speakers)

    return {
        "period_a": retrieve(query, a_chunks, model, k=5, top_n=3) if a_chunks else [],
        "period_b": retrieve(query, b_chunks, model, k=5, top_n=3) if b_chunks else [],
    }