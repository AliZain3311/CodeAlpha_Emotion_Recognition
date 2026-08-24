import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("=" * 70)
print("RAVDESS EMOTION RECOGNITION - SVM HYPERPARAMETER TUNING")
print("=" * 70)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_data = pd.read_csv("data/train_features.csv")

print(f"Training dataset shape: {train_data.shape}")

# ============================================================
# PREPARE TRAINING FEATURES
# ============================================================

X_train = train_data.drop(
    columns=["emotion", "actor", "filename"]
)

y_train = train_data["emotion"]

print(f"Training features shape: {X_train.shape}")

# ============================================================
# SVM PIPELINE
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["rbf", "linear"],
    "svm__gamma": ["scale", "auto"],
    "svm__class_weight": [None, "balanced"]
}

# ============================================================
# GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("STARTING GRID SEARCH")
print("=" * 70)

print("\nTesting SVM hyperparameter combinations...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ============================================================
# BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST SVM PARAMETERS")
print("=" * 70)

print(grid_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(f"{grid_search.best_score_:.4f}")

# ============================================================
# SAVE TUNED MODEL
# ============================================================

best_model = grid_search.best_estimator_

joblib.dump(
    best_model,
    "models/tuned_svm.joblib"
)

print("\nTuned SVM saved to:")
print("models/tuned_svm.joblib")

# ============================================================
# LOAD TEST DATA
# ============================================================

test_data = pd.read_csv("data/test_features.csv")

X_test = test_data.drop(
    columns=["emotion", "actor", "filename"]
)

y_test = test_data["emotion"]

# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TUNED SVM TEST PERFORMANCE")
print("=" * 70)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

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

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("TUNED SVM CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Baseline SVM", "Tuned SVM"],
    "Accuracy": [0.4767, accuracy],
    "Precision": [0.4939, precision],
    "Recall": [0.4767, recall],
    "F1 Score": [0.4754, f1]
})

results.to_csv(
    "data/svm_tuning_results.csv",
    index=False
)

print("\nComparison saved to:")
print("data/svm_tuning_results.csv")

print("\n" + "=" * 70)
print("SVM HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)