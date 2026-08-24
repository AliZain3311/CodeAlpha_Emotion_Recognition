import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/train_features.csv"
TEST_FILE = "data/test_features.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("RAVDESS - LEAKAGE-SAFE FEATURE SELECTION + SVM TUNING")
print("=" * 75)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset shape: {train_df.shape}")
print(f"Testing dataset shape:  {test_df.shape}")


# ============================================================
# PREPARE FEATURES
# ============================================================

DROP_COLUMNS = [
    "emotion",
    "actor",
    "filename"
]

X_train = train_df.drop(columns=DROP_COLUMNS)
X_test = test_df.drop(columns=DROP_COLUMNS)

y_train = train_df["emotion"]
y_test = test_df["emotion"]

groups = train_df["actor"]

print()
print(f"Training features: {X_train.shape[1]}")
print(f"Testing features : {X_test.shape[1]}")


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "selector",
        SelectKBest(
            score_func=f_classif
        )
    ),

    (
        "svm",
        SVC(
            kernel="rbf"
        )
    )
])


# ============================================================
# PARAMETER GRID
# ============================================================

param_grid = {

    "selector__k": [
        75,
        100,
        125,
        150,
        200,
        250,
        "all"
    ],

    "svm__C": [
        1,
        3,
        10,
        30
    ],

    "svm__gamma": [
        "scale",
        0.01,
        0.03
    ],

    "svm__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# GROUP-AWARE CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# GRID SEARCH
# ============================================================

print()
print("=" * 75)
print("STARTING LEAKAGE-SAFE GRID SEARCH")
print("=" * 75)

print()
print("Important:")
print("Test actors are NOT used during model selection.")
print("Feature selection happens inside the CV pipeline.")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid_search.fit(
    X_train,
    y_train,
    groups=groups
)


# ============================================================
# BEST PARAMETERS
# ============================================================

print()
print("=" * 75)
print("BEST PARAMETERS")
print("=" * 75)

print(grid_search.best_params_)

print()
print(
    f"Best CV weighted F1: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

best_model = grid_search.best_estimator_

predictions = best_model.predict(
    X_test
)


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
# TEST RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL UNTOUCHED TEST PERFORMANCE")
print("=" * 75)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "final_svm_pipeline.joblib"
)

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# SAVE CV RESULTS
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)

cv_results = cv_results.sort_values(
    by="rank_test_score"
)

cv_results.to_csv(
    "data/leakage_safe_svm_results.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "CV_F1": grid_search.best_score_,
        "Test_Accuracy": accuracy,
        "Test_Precision": precision,
        "Test_Recall": recall,
        "Test_F1": f1,
        "Best_Params": str(grid_search.best_params_)
    }
])

summary.to_csv(
    "data/final_model_summary.csv",
    index=False
)


# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    f"Final model: {model_path}"
)

print(
    "CV results: data/leakage_safe_svm_results.csv"
)

print(
    "Summary: data/final_model_summary.csv"
)

print()
print("=" * 75)
print("LEAKAGE-SAFE MODEL TRAINING COMPLETED")
print("=" * 75)