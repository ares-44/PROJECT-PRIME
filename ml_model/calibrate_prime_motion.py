import pandas as pd
import numpy as np
import joblib

print("=" * 65)
print("PROJECT PRIME — MOTION MODEL CALIBRATION")
print("=" * 65)

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_random_forest_v2.pkl"

IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_imputer_v2.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

DROP_COLUMNS = [
    "target_north",
    "target_east",
    "target_distance",
    "fix_interval_seconds",
    "gps_fix_index"
]

features = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = imputer.transform(df[features])

pred = model.predict(X)

pred_n = pred[:, 0]
pred_e = pred[:, 1]

true_n = df["target_north"].values
true_e = df["target_east"].values

pred_dist = np.sqrt(
    pred_n ** 2 +
    pred_e ** 2
)

true_dist = np.sqrt(
    true_n ** 2 +
    true_e ** 2
)

# Avoid zero-distance samples
mask = true_dist > 1.0

pred_dist = pred_dist[mask]
true_dist = true_dist[mask]

# =========================================================
# SCALE RATIO
# =========================================================

ratios = pred_dist / true_dist

print("\n===== DISTANCE RATIO =====")

print(
    "Mean ratio:",
    np.mean(ratios)
)

print(
    "Median ratio:",
    np.median(ratios)
)

print(
    "P10 ratio:",
    np.percentile(ratios, 10)
)

print(
    "P90 ratio:",
    np.percentile(ratios, 90)
)

# =========================================================
# LEARN SIMPLE SCALE
# =========================================================

scale = np.sum(
    pred_dist * true_dist
) / np.sum(
    pred_dist ** 2
)

print("\n===== CALIBRATION =====")

print(
    "Optimal distance scale:",
    scale
)

calibrated_dist = pred_dist * scale

error_before = np.abs(
    pred_dist - true_dist
)

error_after = np.abs(
    calibrated_dist - true_dist
)

print("\n===== BEFORE =====")

print(
    "MAE:",
    np.mean(error_before),
    "m"
)

print(
    "Median:",
    np.median(error_before),
    "m"
)

print(
    "P90:",
    np.percentile(error_before, 90),
    "m"
)

print("\n===== AFTER =====")

print(
    "MAE:",
    np.mean(error_after),
    "m"
)

print(
    "Median:",
    np.median(error_after),
    "m"
)

print(
    "P90:",
    np.percentile(error_after, 90),
    "m"
)

print("\n" + "=" * 65)
print("STEP 19 COMPLETE")
print("=" * 65)