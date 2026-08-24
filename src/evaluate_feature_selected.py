import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "data/test_features.csv"

MODEL_FILE = "models/feature_selected_svm.joblib"
SCALER_FILE = "models/feature_selection_scaler.joblib"
SELECTOR_FILE = "models/feature_selector.joblib"

OUTPUT_FILE = "data/feature_selected_confusion_matrix.png"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE-SELECTED SVM - MODEL EVALUATION")
print("=" * 70)

test_df = pd.read_csv(TEST_FILE)

print(
    f"Testing dataset shape: {test_df.shape}"
)

# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test = test_df.drop(
    columns=[
        "emotion",
        "actor",
        "filename"
    ]
)

y_test = test_df["emotion"]

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print("Feature-selected SVM loaded successfully!")

# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(SCALER_FILE)

print("Scaler loaded successfully!")

# ============================================================
# LOAD FEATURE SELECTOR
# ============================================================

selector = joblib.load(SELECTOR_FILE)

print("Feature selector loaded successfully!")

# ============================================================
# TRANSFORM DATA
# ============================================================

X_test_scaled = scaler.transform(X_test)

X_test_selected = selector.transform(
    X_test_scaled
)

print(
    f"Selected test features: {X_test_selected.shape}"
)

# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test_selected
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

# ============================================================
# PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.title(
    "Feature-Selected SVM Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300
)

plt.close()

print(
    f"\nConfusion matrix saved to: {OUTPUT_FILE}"
)

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-SELECTED MODEL EVALUATION COMPLETED")
print("=" * 70)