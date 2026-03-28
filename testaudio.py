import os
# import whisperx
# from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

load_dotenv()

# ---- config ----
device = "cpu"          # use "cpu" if no GPU
compute_type = "int8" # use "int8" for lower memory / CPU
batch_size = 16

print("hello")
# hf_token = os.getenv("HUGGING_FACE_TOKEN")  # set this in your shell/env
# print("Token found:", hf_token is not None)
# print("Token prefix:", hf_token[:6] if hf_token else None)

# audio_dir = "./audio"
# output_dir = "./data/transcripts"
# os.makedirs(output_dir, exist_ok=True)

# # 1) Load WhisperX ASR model
# model = whisperx.load_model("base", device, compute_type=compute_type)

# # 2) Load diarization pipeline
# diarize_model = DiarizationPipeline(token=hf_token, device=device)

# for filename in os.listdir(audio_dir):
#     input_path = os.path.join(audio_dir, filename)

#     if not os.path.isfile(input_path):
#         continue

#     name, _ = os.path.splitext(filename)
#     output_path = os.path.join("test"+ output_dir, name + ".txt")

#     print(f"Processing: {filename}")

#     # Load audio
#     audio = whisperx.load_audio(input_path)

#     # Step A: transcription
#     result = model.transcribe(audio, batch_size=batch_size)

#     # Step B: word alignment
#     model_a, metadata = whisperx.load_align_model(
#         language_code=result["language"],
#         device=device
#     )
#     result = whisperx.align(
#         result["segments"],
#         model_a,
#         metadata,
#         audio,
#         device,
#         return_char_alignments=False
#     )

#     # Step C: diarization
#     diarize_segments = diarize_model(audio)

#     # If you know speaker count, this is better:
#     # diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

#     # Step D: assign speaker labels to words/segments
#     result = whisperx.assign_word_speakers(diarize_segments, result)

#     # Write transcript
#     with open(output_path, "w", encoding="utf-8") as f:
#         for seg in result["segments"]:
#             speaker = seg.get("speaker", "UNKNOWN")
#             text = seg["text"].strip()
#             start = seg.get("start", 0)
#             end = seg.get("end", 0)
#             f.write(f"[{start:.2f} - {end:.2f}] {speaker}: {text}\n")

#     # remove break if you want all files
#     break