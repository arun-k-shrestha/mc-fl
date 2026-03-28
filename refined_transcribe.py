# Install the requests package by executing the command "pip install requests"
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
audio_dir = "./audio"
output_dir = "data/transcripts"

os.makedirs(output_dir, exist_ok=True)  # ensure output folder exists

base_url = "https://api.assemblyai.com"

headers = {
    "authorization": os.getenv("ASSEMBLY_API")
}


for filename in os.listdir(audio_dir):
    input_path = os.path.join(audio_dir, filename)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir,name+".txt")

    print(f"Processing: {filename}")

    with open(input_path, "rb") as f:
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
        output = []
        if transcription_result["status"] == "completed":
            for utt in transcription_result.get("utterances", []):
                start_ms = utt["start"]
                end_ms = utt["end"]
                speaker = utt["speaker"]
                text = utt["text"]
                content =f"{speaker}: {text}"
                output.append(content)
                print(content)
                
            with open(output_path,"w",encoding="utf-8") as f:
                f.write("\n".join(output))

            break

        elif transcription_result["status"] == "error":
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
        
        time.sleep(3)