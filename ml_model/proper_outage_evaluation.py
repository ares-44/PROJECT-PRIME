import pandas as pd
import numpy as np
import joblib

print("=" * 75)
print("PROJECT PRIME — PROPER GNSS OUTAGE EVALUATION V1")
print("=" * 75)

# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_random_forest_v2.pkl"

IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_imputer_v2.pkl"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\proper_outage_evaluation_v1.csv"


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print("\n===== DATA =====")
print("Total intervals:", len(df))


# =========================================================
# FEATURES
# =========================================================

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

intervals = df["fix_interval_seconds"].values


# =========================================================
# CUMULATIVE TRAJECTORY
# =========================================================

true_cum_n = np.concatenate([
    [0],
    np.cumsum(true_n)
])

true_cum_e = np.concatenate([
    [0],
    np.cumsum(true_e)
])

pred_cum_n = np.concatenate([
    [0],
    np.cumsum(pred_n)
])

pred_cum_e = np.concatenate([
    [0],
    np.cumsum(pred_e)
])


# =========================================================
# OUTAGE EVALUATION
# =========================================================

def evaluate_outage(start, number_of_intervals):

    end = start + number_of_intervals

    if end > len(df):
        return None

    start_time = np.sum(intervals[:start])
    end_time = np.sum(intervals[:end])

    outage_duration = end_time - start_time

    # Actual displacement during outage
    actual_n = (
        true_cum_n[end] -
        true_cum_n[start]
    )

    actual_e = (
        true_cum_e[end] -
        true_cum_e[start]
    )

    # PRIME predicted displacement
    predicted_n = (
        pred_cum_n[end] -
        pred_cum_n[start]
    )

    predicted_e = (
        pred_cum_e[end] -
        pred_cum_e[start]
    )

    # Endpoint error
    error = np.sqrt(
        (predicted_n - actual_n) ** 2 +
        (predicted_e - actual_e) ** 2
    )

    actual_distance = np.sqrt(
        actual_n ** 2 +
        actual_e ** 2
    )

    predicted_distance = np.sqrt(
        predicted_n ** 2 +
        predicted_e ** 2
    )

    return {
        "start_interval": start,
        "end_interval": end,
        "outage_duration": outage_duration,
        "actual_distance": actual_distance,
        "predicted_distance": predicted_distance,
        "position_error": error
    }


# =========================================================
# RUN MANY OUTAGES
# =========================================================

outage_lengths = {
    "10_sec": 1,
    "20_sec": 2,
    "30_sec": 3,
    "60_sec": 6
}

all_results = []

print("\n===== EVALUATION =====")

for name, length in outage_lengths.items():

    trials = []

    for start in range(
        0,
        len(df) - length
    ):

        result = evaluate_outage(
            start,
            length
        )

        if result is not None:
            result["outage_class"] = name
            trials.append(result)

            all_results.append(result)

    errors = np.array([
        x["position_error"]
        for x in trials
    ])

    print(f"\n===== {name} =====")

    print(
        "Trials:",
        len(errors)
    )

    print(
        "Mean error:",
        round(np.mean(errors), 2),
        "m"
    )

    print(
        "Median error:",
        round(np.median(errors), 2),
        "m"
    )

    print(
        "P90 error:",
        round(np.percentile(errors, 90), 2),
        "m"
    )

    print(
        "Maximum error:",
        round(np.max(errors), 2),
        "m"
    )


# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== SAVED =====")
print(OUTPUT_PATH)

print("\n" + "=" * 75)
print("STEP 22 COMPLETE")
print("=" * 75)