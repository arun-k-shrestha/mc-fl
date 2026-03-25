from faster_whisper import WhisperModel
import os

model = WhisperModel("base")

audio_dir = "./audio"
output_dir = "data/transcripts"

os.makedirs(output_dir, exist_ok=True)  # ensure output folder exists

for filename in os.listdir(audio_dir):
    input_path = os.path.join(audio_dir, filename)

    # skip non-files (e.g., folders)
    if not os.path.isfile(input_path):
        continue

    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, name + ".txt")

    if os.path.exists(output_path):
        print(f"Skipping (already done): {filename}")
        continue

    print(f"Processing: {filename}")

    segments, info = model.transcribe(input_path)

    output = [segment.text for segment in segments]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))