import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
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
DATA_DIR = "data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("RAVDESS EMOTION RECOGNITION")
print("ADVANCED LEAKAGE-FREE MODEL OPTIMIZATION")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

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

X_train = train_df.drop(columns=DROP_COLUMNS).copy()
X_test = test_df.drop(columns=DROP_COLUMNS).copy()

y_train = train_df["emotion"]
y_test = test_df["emotion"]

groups = train_df["actor"]


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

train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


print()
print(f"Number of input features: {X_train.shape[1]}")
print(f"Training actors: {groups.nunique()}")
print(f"Testing actors: {test_df['actor'].nunique()}")


# ============================================================
# ACTOR OVERLAP CHECK
# ============================================================

train_actors = set(train_df["actor"])
test_actors = set(test_df["actor"])

overlap = train_actors.intersection(test_actors)

print()
print("=" * 75)
print("ACTOR LEAKAGE CHECK")
print("=" * 75)

if overlap:
    print("WARNING: Actor overlap detected!")
    print(overlap)
else:
    print("PASS: No actor overlap detected.")
    print("Training and testing actors are completely separate.")


# ============================================================
# BUILD PIPELINE
# ============================================================

pipeline = Pipeline([
    
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
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# HYPERPARAMETER SEARCH
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
        "auto"
    ],

    "svm__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# ACTOR-AWARE CROSS VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=5
)


# ============================================================
# GRID SEARCH
# ============================================================

print()
print("=" * 75)
print("STARTING LEAKAGE-FREE GRID SEARCH")
print("=" * 75)

print()
print("Important:")
print("The test set is NOT used during model selection.")
print("Feature selection happens inside each CV fold.")
print("Actors are kept separate between CV folds.")

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="f1_weighted",

    cv=cv,

    n_jobs=-1,

    verbose=1,

    return_train_score=True
)


# ============================================================
# TRAIN GRID SEARCH
# ============================================================

print()
print("Training optimized model...")

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

print(
    grid_search.best_params_
)

print()
print(
    f"Best Actor-Aware CV F1 Score: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

print()
print("The test set was kept completely untouched")
print("during feature selection and hyperparameter tuning.")


test_predictions = best_model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)


print()
print("-" * 50)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Weighted Precision: {precision:.4f}"
)

print(
    f"Weighted Recall   : {recall:.4f}"
)

print(
    f"Weighted F1       : {f1:.4f}"
)

print(
    f"Macro F1          : {macro_f1:.4f}"
)

print("-" * 50)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
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

plt.figure(
    figsize=(10, 8)
)

plt.imshow(cm)

plt.title(
    "RAVDESS Emotion Recognition - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(labels)),
    labels,
    rotation=45
)

plt.yticks(
    range(len(labels)),
    labels
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "True Emotion"
)

for i in range(len(labels)):

    for j in range(len(labels)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_path = os.path.join(
    DATA_DIR,
    "advanced_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    f"Confusion matrix saved to: "
    f"{confusion_path}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "advanced_emotion_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)

print()
print(
    f"Optimized model saved to: "
    f"{model_path}"
)


# ============================================================
# SAVE GRID SEARCH RESULTS
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)

cv_results = cv_results.sort_values(
    by="rank_test_score"
)

results_path = os.path.join(
    DATA_DIR,
    "advanced_model_cv_results.csv"
)

cv_results.to_csv(
    results_path,
    index=False
)

print(
    f"CV results saved to: "
    f"{results_path}"
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics = pd.DataFrame([
    {
        "Model": "Advanced Actor-Aware SVM",
        "CV Weighted F1": grid_search.best_score_,
        "Test Accuracy": accuracy,
        "Test Weighted Precision": precision,
        "Test Weighted Recall": recall,
        "Test Weighted F1": f1,
        "Test Macro F1": macro_f1
    }
])

metrics_path = os.path.join(
    DATA_DIR,
    "advanced_model_results.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print(
    f"Final metrics saved to: "
    f"{metrics_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("ADVANCED MODEL TRAINING COMPLETED")
print("=" * 75)

print()
print("BEST PARAMETERS:")
print(grid_search.best_params_)

print()
print(
    f"Actor-Aware CV F1 : "
    f"{grid_search.best_score_:.4f}"
)

print(
    f"Final Test Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Final Test F1      : "
    f"{f1:.4f}"
)

print(
    f"Final Test Macro F1: "
    f"{macro_f1:.4f}"
)

print()
print("Saved files:")

print(
    "models/advanced_emotion_model.joblib"
)

print(
    "data/advanced_model_cv_results.csv"
)

print(
    "data/advanced_model_results.csv"
)

print(
    "data/advanced_confusion_matrix.png"
)

print()
print("=" * 75)