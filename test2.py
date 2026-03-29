import json
from pathlib import Path
from datetime import datetime

INPUT_DIR = Path("data/transcripts")

for file in INPUT_DIR.glob("*.json"):
    with open(file, "r") as r:
        content = json.load(r)

    if "published" in content:
        dt = datetime.strptime(content["published"], "%a, %d %b %Y %H:%M:%S %z")
        content["published"] = dt.strftime("%A, %d %b %Y")

    with open(file, "w") as f:
        json.dump(content, f, indent=2)