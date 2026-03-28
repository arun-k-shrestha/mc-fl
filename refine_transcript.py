import os
name_list = {
    "A": "Mike Caughlan",
    "B": "Shawn Bingham",
    "C": "",
    "D": "",
    "E": ""
}

count = 0

input_dir = "data/transcripts"
output_dir = "data/speaker_diarization"
os.makedirs(output_dir, exist_ok=True)  # ensure output folder exists

filename = "March WASDE Few changes but pay attention.txt"
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
        print(new_line)
        output.append(new_line)

        if count == 4:
            break
        count += 1
    

with open(output_path,"w",encoding="utf-8") as f:
    f.write("\n".join(output))