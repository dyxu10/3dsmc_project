input_path = "flame_mediapipe_landmarks.txt"
output_path = ("flame_mediapipe_landmarks_.txt")

with open(input_path, "r") as f:
    lines = f.readlines()

with open(output_path, "w") as f:
    for line in lines:
        parts = line.strip().split()
        f.write(" ".join(parts[1:]) + "\n")