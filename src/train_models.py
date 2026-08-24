import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================================
# CONFIGURATION
# ==========================================

TRAIN_FILE = "data/train_features.csv"
TEST_FILE = "data/test_features.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("=" * 60)
print("RAVDESS EMOTION RECOGNITION - MODEL TRAINING")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset shape: {train_df.shape}")
print(f"Testing dataset shape:  {test_df.shape}")


# ==========================================
# TARGET
# ==========================================

TARGET = "emotion"

# Columns that should NOT be used as features
DROP_COLUMNS = [
    "emotion",
    "actor",
    "filename"
]

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=DROP_COLUMNS
)

y_test = test_df[TARGET]


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# Use training medians for missing values
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


# ==========================================
# FEATURE SCALING
# ==========================================

print()
print("Applying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    f"Scaled training shape: {X_train_scaled.shape}"
)

print(
    f"Scaled testing shape:  {X_test_scaled.shape}"
)


# ==========================================
# DEFINE MODELS
# ==========================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ),

    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True
    )
}


# ==========================================
# TRAIN MODELS
# ==========================================

results = []

trained_models = {}

print()
print("=" * 60)
print("TRAINING MODELS")
print("=" * 60)

for model_name, model in models.items():

    print()
    print(f"Training: {model_name}")

    # --------------------------------------
    # Train
    # --------------------------------------

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------
    # Prediction
    # --------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    # --------------------------------------
    # Metrics
    # --------------------------------------

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

    # --------------------------------------
    # Store results
    # --------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = model

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )


# ==========================================
# RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print()
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print()
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Best model: {best_model_name}"
)

print(
    f"Best F1 Score: "
    f"{results_df.iloc[0]['F1 Score']:.4f}"
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

scaler_path = os.path.join(
    MODEL_DIR,
    "scaler.joblib"
)

joblib.dump(
    best_model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

print()
print("Saved best model:")
print(model_path)

print()
print("Saved scaler:")
print(scaler_path)


# ==========================================
# SAVE MODEL COMPARISON
# ==========================================

results_path = "data/model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print()
print(
    f"Model comparison saved to: {results_path}"
)


# ==========================================
# BEST MODEL CLASSIFICATION REPORT
# ==========================================

best_predictions = best_model.predict(
    X_test_scaled
)

print()
print("=" * 60)
print("BEST MODEL CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)

print()
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)