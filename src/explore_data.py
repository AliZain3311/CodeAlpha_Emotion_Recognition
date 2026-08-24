import os
from collections import Counter

import librosa


# ==========================================
# RAVDESS EMOTION MAPPING
# ==========================================

emotion_mapping = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ==========================================
# DATASET PATH
# ==========================================

dataset_path = os.path.join(
    "data",
    "Audio_Speech_Actors_01-24"
)


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    print("Expected path:")
    print(dataset_path)
    raise SystemExit


print("==========================================")
print("RAVDESS DATASET EXPLORATION")
print("==========================================")

print("Dataset path:")
print(dataset_path)


# ==========================================
# FIND WAV FILES
# ==========================================

audio_files = []

for root, directories, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith(".wav"):

            audio_files.append(
                os.path.join(root, file)
            )


print("\nTotal audio files:", len(audio_files))


# ==========================================
# ACTOR COUNT
# ==========================================

actors = set()

for file_path in audio_files:

    actor_folder = os.path.basename(
        os.path.dirname(file_path)
    )

    if actor_folder.startswith("Actor_"):

        actors.add(actor_folder)


print("Total actors:", len(actors))


# ==========================================
# EXTRACT EMOTION LABELS
# ==========================================

emotion_counter = Counter()

invalid_files = []


for file_path in audio_files:

    filename = os.path.basename(file_path)

    parts = filename.replace(".wav", "").split("-")

    if len(parts) != 7:

        invalid_files.append(filename)

        continue

    emotion_code = parts[2]

    if emotion_code in emotion_mapping:

        emotion = emotion_mapping[emotion_code]

        emotion_counter[emotion] += 1

    else:

        invalid_files.append(filename)


# ==========================================
# EMOTION DISTRIBUTION
# ==========================================

print("\n========== EMOTION DISTRIBUTION ==========")

for emotion, count in sorted(
    emotion_counter.items()
):

    print(f"{emotion:12s}: {count}")


# ==========================================
# INVALID FILES
# ==========================================

print("\nInvalid files:", len(invalid_files))


# ==========================================
# INSPECT FIRST AUDIO FILE
# ==========================================

if len(audio_files) > 0:

    first_file = audio_files[0]

    print("\n========== FIRST AUDIO FILE ==========")

    print("File:")
    print(first_file)

    try:

        audio, sample_rate = librosa.load(
            first_file,
            sr=None
        )

        duration = len(audio) / sample_rate

        print("Sample rate:", sample_rate)
        print("Audio samples:", len(audio))
        print("Duration:", round(duration, 2), "seconds")

    except Exception as error:

        print("Could not load audio file.")
        print("Error:", error)


# ==========================================
# FINISHED
# ==========================================

print("\n==========================================")
print("Dataset exploration completed successfully!")
print("==========================================")