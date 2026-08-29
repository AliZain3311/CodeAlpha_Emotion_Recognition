import sys
import os
import joblib
import numpy as np
import librosa


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/advanced_emotion_model.joblib"

SAMPLE_RATE = 22050

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]


# ============================================================
# FEATURE HELPER
# ============================================================

def add_statistics(features, values):

    features.append(np.mean(values))
    features.append(np.std(values))
    features.append(np.min(values))
    features.append(np.max(values))


# ============================================================
# ADVANCED FEATURE EXTRACTION
# MUST MATCH TRAINING EXACTLY
# ============================================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    features = []

    # --------------------------------------------------------
    # 1. MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    features.extend(
        np.mean(mfcc, axis=1)
    )

    features.extend(
        np.std(mfcc, axis=1)
    )


    # --------------------------------------------------------
    # 2. DELTA MFCC
    # --------------------------------------------------------

    delta_mfcc = librosa.feature.delta(mfcc)

    features.extend(
        np.mean(delta_mfcc, axis=1)
    )

    features.extend(
        np.std(delta_mfcc, axis=1)
    )


    # --------------------------------------------------------
    # 3. DELTA-2 MFCC
    # --------------------------------------------------------

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.extend(
        np.mean(delta2_mfcc, axis=1)
    )

    features.extend(
        np.std(delta2_mfcc, axis=1)
    )


    # --------------------------------------------------------
    # 4. CHROMA
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    features.extend(
        np.mean(chroma, axis=1)
    )

    features.extend(
        np.std(chroma, axis=1)
    )


    # --------------------------------------------------------
    # 5. SPECTRAL CONTRAST
    # --------------------------------------------------------

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )

    features.extend(
        np.mean(
            spectral_contrast,
            axis=1
        )
    )

    features.extend(
        np.std(
            spectral_contrast,
            axis=1
        )
    )


    # --------------------------------------------------------
    # 6. ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    add_statistics(
        features,
        zcr.flatten()
    )


    # --------------------------------------------------------
    # 7. RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    add_statistics(
        features,
        rms.flatten()
    )


    # --------------------------------------------------------
    # 8. SPECTRAL CENTROID
    # --------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    add_statistics(
        features,
        spectral_centroid.flatten()
    )


    # --------------------------------------------------------
    # 9. SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    add_statistics(
        features,
        spectral_bandwidth.flatten()
    )


    # --------------------------------------------------------
    # 10. SPECTRAL ROLLOFF
    # --------------------------------------------------------

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    add_statistics(
        features,
        spectral_rolloff.flatten()
    )


    # --------------------------------------------------------
    # 11. PITCH / FUNDAMENTAL FREQUENCY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 12. VOICED RATIO
    # --------------------------------------------------------

    voiced_ratio = np.mean(
        voiced_flag
    )

    features.append(
        voiced_ratio
    )


    # --------------------------------------------------------
    # 13. TEMPO
    # --------------------------------------------------------

    tempo, _ = librosa.beat.beat_track(
        y=audio,
        sr=sr
    )

    features.append(
        float(
            np.asarray(tempo).reshape(-1)[0]
        )
    )


    # --------------------------------------------------------
    # CLEAN FEATURES
    # --------------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float64
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


# ============================================================
# PREDICTION
# ============================================================

def predict_emotion(file_path):

    print()
    print("=" * 65)
    print("RAVDESS SPEECH EMOTION RECOGNITION")
    print("=" * 65)

    print()
    print("Audio file:")
    print(file_path)

    # --------------------------------------------------------
    # Check audio
    # --------------------------------------------------------

    if not os.path.isfile(file_path):

        print()
        print("ERROR: Audio file not found!")
        print()
        return


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Advanced emotion model loaded successfully!")


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    print()
    print("Extracting advanced audio features...")

    features = extract_features(
        file_path
    )

    print(
        f"Extracted feature count: {len(features)}"
    )


    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    expected_features = 307

    if len(features) != expected_features:

        print()
        print("=" * 65)
        print("PREDICTION ERROR")
        print("=" * 65)

        print(
            f"Expected {expected_features} features, "
            f"but extracted {len(features)}"
        )

        return


    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    X = features.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X
    )[0]


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X
    )[0]

    probability_dict = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    sorted_probabilities = sorted(
        probability_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("PREDICTION RESULT")
    print("=" * 65)

    print()
    print(
        f"Predicted Emotion : {prediction.upper()}"
    )

    print(
        f"Confidence         : "
        f"{probability_dict[prediction] * 100:.2f}%"
    )

    print()
    print("Top Emotion Probabilities:")
    print("-" * 40)

    for emotion, probability in sorted_probabilities:

        print(
            f"{emotion:<12} : "
            f"{probability * 100:6.2f}%"
        )

    print()
    print("=" * 65)
    print("PREDICTION COMPLETED")
    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("=" * 65)
        print("USAGE")
        print("=" * 65)

        print()
        print(
            'python src\\predict.py "path\\to\\audio.wav"'
        )

        print()
        print("Example:")

        print(
            'python src\\predict.py '
            '"data\\Audio_Speech_Actors_01-24\\Actor_20\\03-01-03-01-01-01-20.wav"'
        )

        sys.exit(1)


    audio_path = sys.argv[1]

    predict_emotion(
        audio_path
    )