from faster_whisper import WhisperModel
import os

model = WhisperModel("base")  # options: tiny, base, small, medium, large-v3


for i in os.listdir("./audio"): #audio is a folder
    print("./audio/"+i)
    # segments, info = model.transcribe("./audio/Energy update on a perfectly timed Friday the 13th.mp3")
    segments, info = model.transcribe("./audio/"+i)

    # print("Detected language:", info.language)
    output = []
    for segment in segments:
        output.append(segment.text)

    with open("./audioText" + i + ".txt","w") as f:
        f.write("\n".join(output))
    
    # open adudioText folder:
    #     add output
    # output = []


    # model = WhisperModel(
    #     "base",
    #     device="cpu",        # or "cuda"
    #     compute_type="int8"  # good for CPU
    # )

    # segments, info = model.transcribe(
    #     "audio.mp3",
    #     beam_size=5
    # )

