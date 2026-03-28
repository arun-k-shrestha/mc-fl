name_list = {
    "A": "Mike Caughlan",
    "B": "Shawn Bingham",
    "C": "",
    "D": "",
    "E": ""
}

count = 0

with open("new_audio_txt.txt", "r") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue  # skip empty lines

        key = line[0].upper()
        name = name_list.get(key, key)  # fallback if key not found

        new_line = name + line[1:]
        print(new_line)

        if count == 4:
            break
        count += 1