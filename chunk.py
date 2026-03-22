from pathlib import Path
import json
from transformers import AutoTokenizer

#INPUT_PATH= Path("audioTextMarch WASDE Few changes but pay attention.mp3.txt") 

CHUNK_SIZE = 500
OVERLAP_TOKENS = 70

INPUT_DIR = Path("data/transcripts")
OUTPUT_FILE = Path("data/chunks/chunks.jsonl")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_sentences(sentences):
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        t = count_tokens(sentence)

        if current and current_tokens + t > CHUNK_SIZE:
            chunks.append({
                "text": " ".join(current),
                "token_count": current_tokens,
            })

            # overlap: reuse last sentences
            overlap = []
            overlap_tokens = 0
            for s in reversed(current):
                print(s)
                ts = count_tokens(s)
                overlap.append(s)
                overlap_tokens += ts
                if overlap_tokens >= OVERLAP_TOKENS:
                    break

            current = overlap
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += t

    if current:
        chunks.append({
            "text": " ".join(current),
            "token_count": current_tokens,
        })
    return chunks

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True) # creates a folder if it doesn't exist

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
        for file in INPUT_DIR.glob("*.txt"):
            text = file.read_text(encoding="utf_8").strip()
            if not text:
                continue
        
            sentences = [line.strip() for line in text.splitlines() if line.strip()]
            chunks = chunk_sentences(sentences)

            for i, chunk in enumerate(chunks,1):
                out_f.write(json.dumps({
                    "chunk_id": f"{file.stem}_{i:04d}",
                    "source_file": file.name,
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                })+ "\n")


if __name__ == "__main__":
    main()
