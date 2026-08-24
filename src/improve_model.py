import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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

TRAIN_FILE = "data/train_features.csv"
TEST_FILE = "data/test_features.csv"

MODEL_DIR = "models"
RESULTS_FILE = "data/improved_model_results.csv"
CONFUSION_FILE = "data/improved_confusion_matrix.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("RAVDESS EMOTION RECOGNITION - ADVANCED MODEL OPTIMIZATION")
print("=" * 75)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset shape: {train_df.shape}")
print(f"Testing dataset shape:  {test_df.shape}")

# ============================================================
# PREPARE DATA
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

print()
print(f"Training features: {X_train.shape[1]}")
print(f"Testing features:  {X_test.shape[1]}")

# ============================================================
# HANDLE INVALID VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

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
            probability=True,
            random_state=RANDOM_STATE
        )
    )
])

# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

print()
print("=" * 75)
print("STARTING ADVANCED GRID SEARCH")
print("=" * 75)

param_grid = {

    "selector__k": [
        75,
        100,
        125,
        150,
        200,
        "all"
    ],

    "svm__C": [
        0.1,
        1,
        10,
        50
    ],

    "svm__gamma": [
        "scale",
        "auto"
    ],

    "svm__kernel": [
        "rbf"
    ],

    "svm__class_weight": [
        None,
        "balanced"
    ]
}

total_combinations = (
    len(param_grid["selector__k"])
    * len(param_grid["svm__C"])
    * len(param_grid["svm__gamma"])
    * len(param_grid["svm__kernel"])
    * len(param_grid["svm__class_weight"])
)

print(
    f"Testing approximately "
    f"{total_combinations} parameter combinations"
)

print("5-fold cross-validation will be used.")
print("Test set will remain completely untouched during tuning.")

# ============================================================
# GRID SEARCH
# ============================================================

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
    y_train
)

# ============================================================
# BEST PARAMETERS
# ============================================================

print()
print("=" * 75)
print("BEST HYPERPARAMETERS")
print("=" * 75)

print(
    grid_search.best_params_
)

print()
print(
    f"Best Cross-Validation Weighted F1: "
    f"{grid_search.best_score_:.4f}"
)

# ============================================================
# CV RESULTS
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)

cv_results = cv_results.sort_values(
    by="mean_test_score",
    ascending=False
)

cv_results[
    [
        "param_selector__k",
        "param_svm__C",
        "param_svm__gamma",
        "param_svm__class_weight",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].to_csv(
    RESULTS_FILE,
    index=False
)

print()
print(
    f"Cross-validation results saved to: "
    f"{RESULTS_FILE}"
)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 75)
print("FINAL UNTOUCHED TEST SET EVALUATION")
print("=" * 75)

best_model = grid_search.best_estimator_

test_predictions = best_model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

print()
print(
    f"Accuracy : {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall   : {test_recall:.4f}"
)

print(
    f"Weighted F1: {test_f1:.4f}"
)

print(
    f"Macro F1   : {test_macro_f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("FINAL CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    test_predictions,
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    CONFUSION_FILE
)

print(
    f"Confusion matrix saved to: "
    f"{CONFUSION_FILE}"
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_emotion_model.joblib"
)

joblib.dump(
    best_model,
    final_model_path
)

print()
print("=" * 75)
print("FINAL MODEL SAVED")
print("=" * 75)

print(
    final_model_path
)

# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = pd.DataFrame([
    {
        "Model": "Optimized SVM",
        "CV Weighted F1": grid_search.best_score_,
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test Weighted F1": test_f1,
        "Test Macro F1": test_macro_f1
    }
])

final_metrics.to_csv(
    "data/final_model_metrics.csv",
    index=False
)

print(
    "Final metrics saved to: "
    "data/final_model_metrics.csv"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("OPTIMIZATION COMPLETED")
print("=" * 75)

print(
    f"Best CV F1       : {grid_search.best_score_:.4f}"
)

print(
    f"Final Test Acc.  : {test_accuracy:.4f}"
)

print(
    f"Final Test F1    : {test_f1:.4f}"
)

print(
    f"Final Macro F1   : {test_macro_f1:.4f}"
)

print()
print("IMPORTANT:")
print(
    "The final test set was not used during hyperparameter "
    "selection."
)

print("=" * 75)