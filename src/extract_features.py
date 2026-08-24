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
# HELPER FUNCTION
# ==========================================

def add_statistics(features, values):

    features.append(np.mean(values))
    features.append(np.std(values))
    features.append(np.min(values))
    features.append(np.max(values))


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    features = []

    # ======================================
    # 1. MFCC
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
    # 2. DELTA MFCC
    # ======================================

    delta_mfcc = librosa.feature.delta(mfcc)

    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    features.extend(delta_mean)
    features.extend(delta_std)


    # ======================================
    # 3. DELTA-2 MFCC
    # ======================================

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)

    features.extend(delta2_mean)
    features.extend(delta2_std)


    # ======================================
    # 4. CHROMA
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
    # 5. SPECTRAL CONTRAST
    # ======================================

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )

    contrast_mean = np.mean(
        spectral_contrast,
        axis=1
    )

    contrast_std = np.std(
        spectral_contrast,
        axis=1
    )

    features.extend(contrast_mean)
    features.extend(contrast_std)


    # ======================================
    # 6. ZERO CROSSING RATE
    # ======================================

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    add_statistics(
        features,
        zcr.flatten()
    )


    # ======================================
    # 7. RMS ENERGY
    # ======================================

    rms = librosa.feature.rms(
        y=audio
    )

    add_statistics(
        features,
        rms.flatten()
    )


    # ======================================
    # 8. SPECTRAL CENTROID
    # ======================================

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    add_statistics(
        features,
        spectral_centroid.flatten()
    )


    # ======================================
    # 9. SPECTRAL BANDWIDTH
    # ======================================

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    add_statistics(
        features,
        spectral_bandwidth.flatten()
    )


    # ======================================
    # 10. SPECTRAL ROLLOFF
    # ======================================

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    add_statistics(
        features,
        spectral_rolloff.flatten()
    )


    # ======================================
    # 11. PITCH / FUNDAMENTAL FREQUENCY
    # ======================================

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    f0_valid = f0[~np.isnan(f0)]

    if len(f0_valid) > 0:

        add_statistics(
            features,
            f0_valid
        )

        features.append(
            np.percentile(f0_valid, 25)
        )

        features.append(
            np.percentile(f0_valid, 50)
        )

        features.append(
            np.percentile(f0_valid, 75)
        )

    else:

        features.extend([
            0, 0, 0, 0,
            0, 0, 0
        ])


    # ======================================
    # 12. VOICED RATIO
    # ======================================

    voiced_ratio = np.mean(
        voiced_flag
    )

    features.append(
        voiced_ratio
    )


    # ======================================
    # 13. TEMPO
    # ======================================

    tempo, _ = librosa.beat.beat_track(
        y=audio,
        sr=sr
    )

    features.append(
        float(np.asarray(tempo).reshape(-1)[0])
    )


    return features


# ==========================================
# MAIN
# ==========================================

print("=" * 60)
print("RAVDESS ADVANCED FEATURE EXTRACTION")
print("=" * 60)


audio_files = glob.glob(
    os.path.join(
        DATASET_PATH,
        "Actor_*",
        "*.wav"
    )
)


print(
    f"Total audio files found: {len(audio_files)}"
)


all_features = []


# ==========================================
# PROCESS FILES
# ==========================================

for index, file_path in enumerate(audio_files):

    filename = os.path.basename(
        file_path
    )

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

        features = extract_features(
            file_path
        )

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

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(audio_files)} files"
        )


# ==========================================
# CREATE COLUMN NAMES
# ==========================================

feature_names = []


# MFCC

for i in range(40):

    feature_names.append(
        f"mfcc_mean_{i + 1}"
    )

for i in range(40):

    feature_names.append(
        f"mfcc_std_{i + 1}"
    )


# Delta MFCC

for i in range(40):

    feature_names.append(
        f"delta_mfcc_mean_{i + 1}"
    )

for i in range(40):

    feature_names.append(
        f"delta_mfcc_std_{i + 1}"
    )


# Delta-2 MFCC

for i in range(40):

    feature_names.append(
        f"delta2_mfcc_mean_{i + 1}"
    )

for i in range(40):

    feature_names.append(
        f"delta2_mfcc_std_{i + 1}"
    )


# Chroma

for i in range(12):

    feature_names.append(
        f"chroma_mean_{i + 1}"
    )

for i in range(12):

    feature_names.append(
        f"chroma_std_{i + 1}"
    )


# Spectral Contrast

for i in range(7):

    feature_names.append(
        f"contrast_mean_{i + 1}"
    )

for i in range(7):

    feature_names.append(
        f"contrast_std_{i + 1}"
    )


# Other statistical features

feature_names.extend([
    "zcr_mean",
    "zcr_std",
    "zcr_min",
    "zcr_max",

    "rms_mean",
    "rms_std",
    "rms_min",
    "rms_max",

    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_centroid_min",
    "spectral_centroid_max",

    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_bandwidth_min",
    "spectral_bandwidth_max",

    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_rolloff_min",
    "spectral_rolloff_max",

    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "pitch_25",
    "pitch_median",
    "pitch_75",

    "voiced_ratio",
    "tempo",

    "emotion",
    "actor",
    "filename"
])


# ==========================================
# DATAFRAME
# ==========================================

df = pd.DataFrame(
    all_features,
    columns=feature_names
)


# ==========================================
# REMOVE INVALID VALUES
# ==========================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

df[numeric_columns] = df[
    numeric_columns
].fillna(0)


# ==========================================
# SAVE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# RESULTS
# ==========================================

print()
print("=" * 60)
print("ADVANCED FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print(
    f"Feature dataset shape: {df.shape}"
)

print()
print("Emotion distribution:")

print(
    df["emotion"].value_counts()
)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print("=" * 60)