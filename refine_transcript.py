import os
name_list = {
    "A": "Kevin Combs",
    "B": "Craig Ruffolo",
    "C": "",
    "D": "",
    "E": ""
}

input_dir = "data/transcripts"
output_dir = "data/speaker_diarization"
os.makedirs(output_dir, exist_ok=True)  # ensure output folder exists

filename = "Sugar Mayhem is everywhere.txt"
input_path = os.path.join(input_dir, filename)
output_path = os.path.join(output_dir,filename)

with open(input_path, "r") as f:
    output = []
    for line in f:
        line = line.strip()

        if not line:
            continue  # skip empty lines

        key = line[0].upper()
        name = name_list.get(key, key)  # fallback if key not found

        new_line = name + line[1:]
        output.append(new_line)
    

with open(output_path,"w",encoding="utf-8") as f:
    f.write("\n".join(output))