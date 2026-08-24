import os
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "data/features.csv"

TRAIN_FILE = "data/train_features.csv"
TEST_FILE = "data/test_features.csv"

# Actors 01-19 for training
TRAIN_ACTORS = [
    f"Actor_{i:02d}"
    for i in range(1, 20)
]

# Actors 20-24 for testing
TEST_ACTORS = [
    f"Actor_{i:02d}"
    for i in range(20, 25)
]


# ==========================================
# LOAD FEATURES
# ==========================================

print("=" * 50)
print("RAVDESS TRAIN / TEST PREPARATION")
print("=" * 50)

df = pd.read_csv(INPUT_FILE)

print(f"Original dataset shape: {df.shape}")


# ==========================================
# TRAIN / TEST SPLIT BY ACTOR
# ==========================================

train_df = df[
    df["actor"].isin(TRAIN_ACTORS)
].copy()

test_df = df[
    df["actor"].isin(TEST_ACTORS)
].copy()


# ==========================================
# VERIFY ACTORS
# ==========================================

print()
print("Training actors:")
print(
    sorted(train_df["actor"].unique())
)

print()
print("Testing actors:")
print(
    sorted(test_df["actor"].unique())
)


# ==========================================
# DATASET SHAPES
# ==========================================

print()
print(f"Training dataset shape: {train_df.shape}")
print(f"Testing dataset shape: {test_df.shape}")


# ==========================================
# EMOTION DISTRIBUTION
# ==========================================

print()
print("========== TRAIN EMOTION DISTRIBUTION ==========")

print(
    train_df["emotion"].value_counts()
    .sort_index()
)

print()
print("========== TEST EMOTION DISTRIBUTION ==========")

print(
    test_df["emotion"].value_counts()
    .sort_index()
)


# ==========================================
# CHECK ACTOR OVERLAP
# ==========================================

train_actor_set = set(
    train_df["actor"].unique()
)

test_actor_set = set(
    test_df["actor"].unique()
)

overlap = train_actor_set.intersection(
    test_actor_set
)

print()
print("========== ACTOR OVERLAP CHECK ==========")

if len(overlap) == 0:
    print("No actor overlap detected!")
    print("Train and test actors are completely separate.")
else:
    print("WARNING: Actor overlap detected!")
    print(overlap)


# ==========================================
# SAVE DATASETS
# ==========================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print()
print("=" * 50)
print("TRAIN / TEST PREPARATION COMPLETED")
print("=" * 50)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

print()
print(f"Training file: {TRAIN_FILE}")
print(f"Testing file: {TEST_FILE}")

print("=" * 50)