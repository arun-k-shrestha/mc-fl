import requests
import feedparser
import os
import re
import json
import html

url = "https://feed.podbean.com/mckeanyflavell/feed.xml"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
feed = feedparser.parse(response.content)

folder = "audio"
text_folder = "data/transcripts"
os.makedirs(folder, exist_ok=True)

def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text)

def clean_summary(text):
    if not text:
        return ""

    # Decode HTML entities like &amp;
    text = html.unescape(text)

    # Replace all whitespace/newlines/tabs with single spaces
    text = re.sub(r"\s+", " ", text)

    # Trim extra spaces
    return text.strip()

for entry in feed.entries[:10]:
    title = clean_filename(entry.title)
    if not entry.enclosures:
        continue
    mp3_url = entry.enclosures[0].href

    print(f"Downloading: {title}")

    audio_response = requests.get(mp3_url, headers=headers)

    file_path = os.path.join(folder, f"{title}.mp3")

    with open(file_path, "wb") as f:
        f.write(audio_response.content)
    
    metadata = {
        "title": entry.get("title", ""),
        "published": entry.get("published", ""),
        "summary": clean_summary(entry.get("summary", "")),
    }

    json_path = os.path.join(text_folder, f"{title}.json")
    print("Writing:", os.path.abspath(json_path))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata: {json_path}")




    