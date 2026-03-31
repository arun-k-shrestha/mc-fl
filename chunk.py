from pathlib import Path
import json
import re
from transformers import AutoTokenizer

CHUNK_SIZE = 500
OVERLAP_TOKENS = 70
MAX_TURN_TOKENS = 180

INPUT_DIR = Path("data/speaker_diarization")
OUTPUT_FILE = Path("data/chunks/chunks.jsonl")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def parse_speaker(text: str) -> list[dict]:
    sentences = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if ":" not in line:
            continue

        speaker, content = line.split(":", 1)
        speaker = speaker.strip()
        content = content.strip()

        if content:
            sentences.append({"speaker": speaker, "text": content})

    return sentences


def format_sentence(record: dict) -> str:
    return f'{record["speaker"]}: {record["text"]}'


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_long_record(record: dict) -> list[dict]:
    formatted = format_sentence(record)
    if count_tokens(formatted) <= MAX_TURN_TOKENS:
        return [record]

    speaker = record["speaker"]
    sentence_parts = split_into_sentences(record["text"])

    pieces = []
    current_texts = []

    for sent in sentence_parts:
        trial_text = " ".join(current_texts + [sent])
        trial_record = {"speaker": speaker, "text": trial_text}

        if current_texts and count_tokens(format_sentence(trial_record)) > MAX_TURN_TOKENS:
            pieces.append({"speaker": speaker, "text": " ".join(current_texts)})
            current_texts = [sent]
        else:
            current_texts.append(sent)

    if current_texts:
        pieces.append({"speaker": speaker, "text": " ".join(current_texts)})

    return pieces


def normalize_records(records: list[dict]) -> list[dict]:
    normalized = []
    for record in records:
        normalized.extend(split_long_record(record))
    return normalized


def load_metadata(txt_path):
    print(txt_path)
    meta_path = txt_path.with_suffix(".json")
    print(f"Transcript: {txt_path}")
    print(f"Metadata:   {meta_path}")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def chunk_sentences(sentences):
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        formatted = format_sentence(sentence)
        t = count_tokens(formatted)

        if current and current_tokens + t > CHUNK_SIZE:
            chunk_text = " ".join(format_sentence(r) for r in current)
            chunk_speakers = list(dict.fromkeys(r["speaker"] for r in current))
            chunks.append({
                "text": chunk_text,
                "token_count": current_tokens,
                "speakers": chunk_speakers,
            })

            # overlap: reuse last sentences
            overlap = []
            overlap_tokens = 0
            for s in reversed(current):
                fs = format_sentence(s)
                ts = count_tokens(fs)
                if overlap and overlap_tokens + ts > OVERLAP_TOKENS:
                    break
                overlap.append(s)
                overlap_tokens += ts

            overlap.reverse()
            current = overlap
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += t

    if current:
        chunk_text = " ".join(format_sentence(r) for r in current)
        chunk_speakers = list(dict.fromkeys(r["speaker"] for r in current))
        chunks.append({
            "text": chunk_text,
            "token_count": current_tokens,
            "speakers": chunk_speakers,
        })

    return chunks


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
        for file in INPUT_DIR.glob("*.txt"):
            text = file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            metadata = load_metadata(Path("data/transcripts") / file.stem)

            sentences = parse_speaker(text)
            sentences = normalize_records(sentences)   # minimal added line
            chunks = chunk_sentences(sentences)

            for i, chunk in enumerate(chunks, 1):
                out_f.write(json.dumps({
                    "chunk_id": f"{file.stem} {i:04d}",
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                    "speakers": chunk["speakers"],
                    "title": metadata.get("title"),
                    "day": metadata.get("published"),
                    "summary": metadata.get("summary"),
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()