import os
from collections import Counter

import librosa
import librosa.display
import matplotlib.pyplot as plt


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
# FIND AUDIO FILES
# ==========================================

audio_files = []

for root, directories, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith(".wav"):

            audio_files.append(
                os.path.join(root, file)
            )


print("==========================================")
print("RAVDESS AUDIO EDA")
print("==========================================")

print("Total audio files:", len(audio_files))


# ==========================================
# EXTRACT EMOTIONS
# ==========================================

emotion_counter = Counter()

for file_path in audio_files:

    filename = os.path.basename(file_path)

    parts = filename.replace(".wav", "").split("-")

    if len(parts) == 7:

        emotion_code = parts[2]

        if emotion_code in emotion_mapping:

            emotion = emotion_mapping[emotion_code]

            emotion_counter[emotion] += 1


# ==========================================
# 1. EMOTION DISTRIBUTION
# ==========================================

emotions = list(emotion_counter.keys())
counts = list(emotion_counter.values())

plt.figure(figsize=(10, 6))

plt.bar(emotions, counts)

plt.xlabel("Emotion")
plt.ylabel("Number of Audio Files")
plt.title("RAVDESS Emotion Distribution")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "emotion_distribution.png",
    dpi=300
)

plt.show()


# ==========================================
# SELECT FIRST AUDIO
# ==========================================

if len(audio_files) == 0:

    print("No audio files found!")

    raise SystemExit


audio_file = audio_files[0]


print("\nSelected audio file:")
print(audio_file)


# ==========================================
# LOAD AUDIO
# ==========================================

audio, sample_rate = librosa.load(
    audio_file,
    sr=None
)


print("Sample rate:", sample_rate)

print(
    "Duration:",
    round(len(audio) / sample_rate, 2),
    "seconds"
)


# ==========================================
# 2. WAVEFORM
# ==========================================

plt.figure(figsize=(12, 5))

librosa.display.waveshow(
    audio,
    sr=sample_rate
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.title("Speech Audio Waveform")

plt.tight_layout()

plt.savefig(
    "waveform.png",
    dpi=300
)

plt.show()


# ==========================================
# 3. MFCC
# ==========================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)


print("\nMFCC shape:", mfcc.shape)


plt.figure(figsize=(12, 6))

librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sample_rate
)

plt.colorbar()

plt.xlabel("Time")

plt.ylabel("MFCC Coefficient")

plt.title("MFCC Features")

plt.tight_layout()

plt.savefig(
    "mfcc_visualization.png",
    dpi=300
)

plt.show()


# ==========================================
# FINISHED
# ==========================================

print("\n==========================================")
print("EDA completed successfully!")
print("==========================================")

print("\nGenerated files:")

print("emotion_distribution.png")
print("waveform.png")
print("mfcc_visualization.png")