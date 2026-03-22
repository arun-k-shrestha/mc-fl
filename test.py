import requests
import feedparser
import os
import re

url = "https://feed.podbean.com/mckeanyflavell/feed.xml"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
feed = feedparser.parse(response.content)

folder = "audio"
os.makedirs(folder, exist_ok=True)

def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text)

for entry in feed.entries[:2]:
    title = clean_filename(entry.title)
    if not entry.enclosures:
        continue
    mp3_url = entry.enclosures[0].href

    print(f"Downloading: {title}")

    audio_response = requests.get(mp3_url, headers=headers)

    file_path = os.path.join(folder, f"{title}.mp3")

    with open(file_path, "wb") as f:
        f.write(audio_response.content)