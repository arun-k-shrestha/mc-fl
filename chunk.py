from pathlib import Path
import json
from transformers import AutoTokenizer

#INPUT_PATH= Path("audioTextMarch WASDE Few changes but pay attention.txt") 

CHUNK_SIZE = 500
OVERLAP_TOKENS = 70

INPUT_DIR = Path("data/speaker_diarization")
OUTPUT_FILE = Path("data/chunks/chunks.jsonl")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def parse_speaker(text:str) -> list[dict]:
    """
    Parse transcript lines like:
    Speaker NameA: text
    Speaker NameB: text

    Returns:
        [
            {"speaker": "Mike Caughlan", "text": "..."},
            {"speaker": "Shawn Bingham", "text": "..."},
        ]
    """
    sentences = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if ":" in line:
            speaker, text = line.split(":",1)
            speaker= speaker.strip()
            text = text.strip()
        if text:
            sentences.append({"speaker":speaker, "text":text})
    
    return sentences

def format_sentence(record: dict) -> str:
    return f'{record["speaker"]}: {record["text"]}'

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
                overlap.append(s)
                overlap_tokens += ts
                if overlap_tokens >= OVERLAP_TOKENS:
                    break
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
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True) # creates a folder if it doesn't exist

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
        for file in INPUT_DIR.glob("*.txt"):
            text = file.read_text(encoding="utf_8").strip()
            if not text:
                continue
            
            metadata = load_metadata(Path("data/transcripts") / file.stem)


            sentences = parse_speaker(text)
            chunks = chunk_sentences(sentences)

            for i, chunk in enumerate(chunks,1):
                out_f.write(json.dumps({
                    "chunk_id": f"{file.stem}_{i:04d}",
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                    "speakers": chunk["speakers"],
                    "title": metadata.get("title"),
                    "published": metadata.get("published"),
                    "summary": metadata.get("summary"),

                })+ "\n")
            
            break


if __name__ == "__main__":
    main()
