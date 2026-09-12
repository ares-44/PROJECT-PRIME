import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

# ============================================================
# PROJECT PRIME
# STEP 10 — FIRST ML MODEL
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v2.csv"

MODEL_DIR = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("PROJECT PRIME — FIRST ML MODEL")
print("========================================")

print("\n===== DATA =====")
print("Samples:", len(df))
print("Columns:", len(df.columns))

# ------------------------------------------------------------
# TARGETS
# ------------------------------------------------------------

targets = [
    "target_north",
    "target_east",
    "target_distance"
]

X = df.drop(columns=targets)

y = df[targets]

print("\nFeatures:", X.shape[1])
print("Targets:", y.shape[1])

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "random_forest",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print("\n===== TRAINING =====")

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

predictions = model.predict(X_test)

pred_df = pd.DataFrame(
    predictions,
    columns=targets,
    index=y_test.index
)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

for target in targets:

    actual = y_test[target]
    predicted = pred_df[target]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n-----", target, "-----")

    print(
        "MAE:",
        round(mae, 4),
        "m"
    )

    print(
        "RMSE:",
        round(rmse, 4),
        "m"
    )

    print(
        "R²:",
        round(r2, 4)
    )

# ------------------------------------------------------------
# OVERALL DISTANCE ERROR
# ------------------------------------------------------------

actual_distance = y_test["target_distance"].to_numpy()

predicted_distance = pred_df[
    "target_distance"
].to_numpy()

distance_error = np.abs(
    actual_distance -
    predicted_distance
)

print("\n===== DISTANCE ERROR =====")

print(
    "Mean:",
    np.mean(distance_error),
    "m"
)

print(
    "Median:",
    np.median(distance_error),
    "m"
)

print(
    "P90:",
    np.percentile(distance_error, 90),
    "m"
)

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "prime_random_forest_v1.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\n===== MODEL SAVED =====")

print(model_path)

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

rf = model.named_steps["random_forest"]

importance = rf.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_path = os.path.join(
    MODEL_DIR,
    "prime_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print("\n===== TOP 15 FEATURES =====")

print(
    importance_df.head(15).to_string(
        index=False
    )
)

print("\n===== FEATURE IMPORTANCE SAVED =====")

print(importance_path)

print("\n========================================")
print("STEP 10 COMPLETE")
print("========================================")