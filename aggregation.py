# aggregation.py
from collections import defaultdict
from retrieval import filter_chunks,retrieve


def retrieve_all_episodes(query, chunks, model, speakers=None, top_per_episode=2):
    """For multi-episode questions: get top chunks from EACH episode separately."""
    by_episode = defaultdict(list)
    filtered = filter_chunks(chunks, speakers=speakers)
    for c in filtered:
        by_episode[c.get("title", "unknown")].append(c)

    results = []
    for title, ep_chunks in by_episode.items():
        top = retrieve(query, ep_chunks, model, k=5, top_n=top_per_episode)
        results.extend(top)

    # Sort by date ascending so LLM sees chronological order
    results.sort(key=lambda c: c.get("day", ""))
    return results