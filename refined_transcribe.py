# Install the requests package by executing the command "pip install requests"
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
base_url = "https://api.assemblyai.com"

headers = {
    "authorization": os.getenv("ASSEMBLY_API")
}

with open("test.mp3", "rb") as f:
  response = requests.post(base_url + "/v2/upload",
                          headers=headers,
                          data=f)

audio_url = response.json()["upload_url"]

data = {
    "audio_url": audio_url,
    "language_detection": True,
    "speech_models": ["universal-3-pro", "universal-2"],
    "speaker_labels": True
    # optional:
    # "speakers_expected": 2
}

response = requests.post(
    f"{base_url}/v2/transcript",
    json=data,
    headers=headers
)
response.raise_for_status()

transcript_id = response.json()["id"]
polling_endpoint = f"{base_url}/v2/transcript/{transcript_id}"

while True:
    transcription_result = requests.get(polling_endpoint, headers=headers).json()

    if transcription_result["status"] == "completed":
        for utt in transcription_result.get("utterances", []):
            start_ms = utt["start"]
            end_ms = utt["end"]
            speaker = utt["speaker"]
            text = utt["text"]

            print(f"{speaker}: {text}")
        break

    elif transcription_result["status"] == "error":
        raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

    time.sleep(3)