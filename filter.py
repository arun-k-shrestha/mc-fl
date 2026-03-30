from pathlib import Path
import json

input_dir = Path("data/transcripts")
content = []

for file in sorted(input_dir.glob("*.json")):
    print(file)
    with open(file, "r", encoding="utf-8") as f:
        content.append(json.load(f))  # parse JSON

print(content)


