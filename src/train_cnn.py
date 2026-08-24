import os
import glob
import numpy as np
import librosa
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/Audio_Speech_Actors_01-24"
MODEL_PATH = "models/cnn_emotion_model.keras"
LABEL_PATH = "models/cnn_label_encoder.npy"
CONFUSION_MATRIX_PATH = "cnn_confusion_matrix.png"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 174

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# EMOTION MAPPING
# ============================================================

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


# ============================================================
# ACTOR SPLIT
# ============================================================

TRAIN_ACTORS = [
    f"Actor_{i:02d}"
    for i in range(1, 20)
]

TEST_ACTORS = [
    f"Actor_{i:02d}"
    for i in range(20, 25)
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mfcc(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    # Delta
    delta = librosa.feature.delta(mfcc)

    # Delta-Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # --------------------------------------------------------
    # Normalize each channel
    # --------------------------------------------------------

    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-8
    )

    delta = (
        delta - np.mean(delta)
    ) / (
        np.std(delta) + 1e-8
    )

    delta2 = (
        delta2 - np.mean(delta2)
    ) / (
        np.std(delta2) + 1e-8
    )

    # --------------------------------------------------------
    # Pad / truncate
    # --------------------------------------------------------

    def pad_or_truncate(feature):

        if feature.shape[1] < MAX_FRAMES:

            pad_width = MAX_FRAMES - feature.shape[1]

            feature = np.pad(
                feature,
                (
                    (0, 0),
                    (0, pad_width)
                ),
                mode="constant"
            )

        else:

            feature = feature[:, :MAX_FRAMES]

        return feature

    mfcc = pad_or_truncate(mfcc)
    delta = pad_or_truncate(delta)
    delta2 = pad_or_truncate(delta2)

    # --------------------------------------------------------
    # Stack as 3 channels
    # --------------------------------------------------------

    feature = np.stack(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=-1
    )

    return feature.astype(np.float32)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RAVDESS EMOTION RECOGNITION - CNN TRAINING")
print("=" * 70)

audio_files = glob.glob(
    os.path.join(
        DATASET_PATH,
        "Actor_*",
        "*.wav"
    )
)

print(f"Total audio files found: {len(audio_files)}")


X_train = []
y_train = []

X_test = []
y_test = []


# ============================================================
# PROCESS AUDIO
# ============================================================

for index, file_path in enumerate(audio_files):

    filename = os.path.basename(file_path)

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

        feature = extract_mfcc(file_path)

        if actor in TRAIN_ACTORS:

            X_train.append(feature)
            y_train.append(emotion)

        elif actor in TEST_ACTORS:

            X_test.append(feature)
            y_test.append(emotion)

    except Exception as e:

        print(
            f"Error processing {filename}: {e}"
        )

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(audio_files)} files"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)


print()
print("=" * 70)
print("DATA PREPARATION")
print("=" * 70)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape : {X_test.shape}")


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train
)

y_test_encoded = label_encoder.transform(
    y_test
)

print()
print("Emotion classes:")

for index, emotion in enumerate(
    label_encoder.classes_
):

    print(
        f"{index}: {emotion}"
    )


# ============================================================
# SAVE LABEL ENCODER
# ============================================================

np.save(
    LABEL_PATH,
    label_encoder.classes_
)


# ============================================================
# CNN MODEL
# ============================================================

print()
print("=" * 70)
print("BUILDING CNN MODEL")
print("=" * 70)


model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            N_MFCC,
            MAX_FRAMES,
            3
        )
    ),

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Dropout(
        0.25
    ),

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Dropout(
        0.25
    ),

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    # --------------------------------------------------------
    # Global Pooling
    # --------------------------------------------------------

    tf.keras.layers.GlobalAveragePooling2D(),

    # --------------------------------------------------------
    # Dense layers
    # --------------------------------------------------------

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.40
    ),

    tf.keras.layers.Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )

])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print()
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=1e-6
)


# ============================================================
# TRAIN CNN
# ============================================================

print()
print("=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)


history = model.fit(

    X_train,
    y_train_encoded,

    validation_split=0.15,

    epochs=40,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# TEST PREDICTION
# ============================================================

print()
print("=" * 70)
print("CNN TEST EVALUATION")
print("=" * 70)


probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_encoded,
    predictions
)

precision = precision_score(
    y_test_encoded,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test_encoded,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test_encoded,
    predictions,
    average="weighted",
    zero_division=0
)


print()
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CNN CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test_encoded,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_encoded,
    predictions
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title(
    "Confusion Matrix - CNN Emotion Recognition"
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "Actual Emotion"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200
)

plt.close()


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = {
    "epoch": range(
        1,
        len(history.history["loss"]) + 1
    ),

    "loss": history.history["loss"],

    "accuracy": history.history["accuracy"],

    "val_loss": history.history["val_loss"],

    "val_accuracy": history.history["val_accuracy"]
}

import pandas as pd

history_df = pd.DataFrame(
    history_df
)

history_df.to_csv(
    "data/cnn_training_history.csv",
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("CNN TRAINING COMPLETED")
print("=" * 70)

print(
    f"Model saved to: {MODEL_PATH}"
)

print(
    f"Label encoder saved to: {LABEL_PATH}"
)

print(
    f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}"
)

print(
    "Training history saved to: data/cnn_training_history.csv"
)

print("=" * 70)