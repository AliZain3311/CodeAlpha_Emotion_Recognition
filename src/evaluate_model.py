import pandas as pd
import joblib
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

print("=" * 60)
print("RAVDESS EMOTION RECOGNITION - MODEL EVALUATION")
print("=" * 60)

# ==========================================
# LOAD TEST DATA
# ==========================================

test_data = pd.read_csv("data/test_features.csv")

print(f"Testing dataset shape: {test_data.shape}")

# ==========================================
# PREPARE FEATURES AND TARGET
# ==========================================

X_test = test_data.drop(
    columns=["emotion", "actor", "filename"]
)

y_test = test_data["emotion"]

# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")

print("Best model loaded successfully!")
print("Scaler loaded successfully!")

# ==========================================
# SCALE TEST DATA
# ==========================================

X_test_scaled = scaler.transform(X_test)

print(f"Scaled test shape: {X_test_scaled.shape}")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_scaled)

# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(10, 8))

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
plt.title("Confusion Matrix - SVM Emotion Recognition")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)