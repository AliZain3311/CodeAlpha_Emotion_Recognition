import os
import glob
import numpy as np
import pandas as pd
import librosa

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "data/Audio_Speech_Actors_01-24"
OUTPUT_FILE = "data/features.csv"

SAMPLE_RATE = 22050

# ==========================================
# EMOTION MAPPING
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
# FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    features = []

    # ======================================
    # MFCC
    # ======================================

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features.extend(mfcc_mean)
    features.extend(mfcc_std)

    # ======================================
    # DELTA MFCC
    # ======================================

    delta_mfcc = librosa.feature.delta(mfcc)

    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    features.extend(delta_mean)
    features.extend(delta_std)

    # ======================================
    # CHROMA
    # ======================================

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    features.extend(chroma_mean)
    features.extend(chroma_std)

    # ======================================
    # ZERO CROSSING RATE
    # ======================================

    zcr = librosa.feature.zero_crossing_rate(audio)

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # ======================================
    # RMS ENERGY
    # ======================================

    rms = librosa.feature.rms(y=audio)

    features.append(np.mean(rms))
    features.append(np.std(rms))

    # ======================================
    # SPECTRAL CENTROID
    # ======================================

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))

    # ======================================
    # SPECTRAL BANDWIDTH
    # ======================================

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))

    # ======================================
    # SPECTRAL ROLLOFF
    # ======================================

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    features.append(np.mean(spectral_rolloff))
    features.append(np.std(spectral_rolloff))

    return features


# ==========================================
# MAIN
# ==========================================

print("=" * 50)
print("RAVDESS FEATURE EXTRACTION")
print("=" * 50)

audio_files = glob.glob(
    os.path.join(
        DATASET_PATH,
        "Actor_*",
        "*.wav"
    )
)

print(f"Total audio files found: {len(audio_files)}")

all_features = []

# ==========================================
# PROCESS AUDIO FILES
# ==========================================

for index, file_path in enumerate(audio_files):

    filename = os.path.basename(file_path)

    # Example:
    # 03-01-05-01-02-01-12.wav

    parts = filename.split("-")

    emotion_code = parts[2]

    emotion = emotion_mapping.get(
        emotion_code,
        "unknown"
    )

    actor = os.path.basename(
        os.path.dirname(file_path)
    )

    try:

        features = extract_features(file_path)

        row = features + [
            emotion,
            actor,
            filename
        ]

        all_features.append(row)

    except Exception as e:

        print(
            f"Error processing {filename}: {e}"
        )

    # Progress

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(audio_files)} files"
        )


# ==========================================
# CREATE COLUMN NAMES
# ==========================================

feature_names = []

# MFCC mean

for i in range(40):
    feature_names.append(
        f"mfcc_mean_{i + 1}"
    )

# MFCC std

for i in range(40):
    feature_names.append(
        f"mfcc_std_{i + 1}"
    )

# Delta MFCC mean

for i in range(40):
    feature_names.append(
        f"delta_mfcc_mean_{i + 1}"
    )

# Delta MFCC std

for i in range(40):
    feature_names.append(
        f"delta_mfcc_std_{i + 1}"
    )

# Chroma mean

for i in range(12):
    feature_names.append(
        f"chroma_mean_{i + 1}"
    )

# Chroma std

for i in range(12):
    feature_names.append(
        f"chroma_std_{i + 1}"
    )

# Other features

feature_names.extend([
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "emotion",
    "actor",
    "filename"
])

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(
    all_features,
    columns=feature_names
)

# ==========================================
# SAVE FEATURES
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================
# RESULTS
# ==========================================

print()
print("=" * 50)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 50)

print(f"Feature dataset shape: {df.shape}")

print()
print("Emotion distribution:")

print(
    df["emotion"].value_counts()
)

print()
print(f"Saved to: {OUTPUT_FILE}")

print("=" * 50)