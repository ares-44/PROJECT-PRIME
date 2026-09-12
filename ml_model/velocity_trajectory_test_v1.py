import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("PROJECT PRIME — VELOCITY TRAJECTORY TEST V1")
print("=" * 70)

# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_velocity_dataset_v1.csv"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_velocity_rf_v1.pkl"

IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_velocity_imputer_v1.pkl"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\velocity_trajectory_test_v1.csv"


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))


# =========================================================
# FEATURES
# =========================================================

DROP_COLUMNS = [
    "target_velocity_north",
    "target_velocity_east",
    "target_velocity",
    "interval_seconds",
    "gps_fix_index"
]

features = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = imputer.transform(
    df[features]
)

# =========================================================
# PREDICT VELOCITY
# =========================================================

pred = model.predict(X)

pred_vn = pred[:, 0]
pred_ve = pred[:, 1]

true_vn = df[
    "target_velocity_north"
].values

true_ve = df[
    "target_velocity_east"
].values

intervals = df[
    "interval_seconds"
].values


# =========================================================
# CHRONOLOGICAL TEST REGION
# =========================================================

n = len(df)

test_start = int(n * 0.85)

print("\n===== TEST REGION =====")
print("Start index:", test_start)
print("Test samples:", n - test_start)


# =========================================================
# OUTAGE SIMULATION
# =========================================================

def evaluate_outage(
    start,
    number_of_intervals
):

    end = start + number_of_intervals

    if end > n:
        return None

    # -----------------------------------------------------
    # Predicted displacement from velocity
    # -----------------------------------------------------

    predicted_n = np.sum(
        pred_vn[start:end]
        *
        intervals[start:end]
    )

    predicted_e = np.sum(
        pred_ve[start:end]
        *
        intervals[start:end]
    )

    # -----------------------------------------------------
    # Actual displacement
    # -----------------------------------------------------

    actual_n = np.sum(
        true_vn[start:end]
        *
        intervals[start:end]
    )

    actual_e = np.sum(
        true_ve[start:end]
        *
        intervals[start:end]
    )

    # -----------------------------------------------------
    # Error
    # -----------------------------------------------------

    error = np.sqrt(
        (predicted_n - actual_n) ** 2
        +
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

    duration = np.sum(
        intervals[start:end]
    )

    return {
        "start": start,
        "duration": duration,
        "actual_distance": actual_distance,
        "predicted_distance": predicted_distance,
        "error": error
    }


# =========================================================
# TEST
# =========================================================

outages = {
    "10_sec": 1,
    "20_sec": 2,
    "30_sec": 3,
    "60_sec": 6
}

results = []


print("\n===== OUTAGE RESULTS =====")


for name, length in outages.items():

    trials = []

    for start in range(
        test_start,
        n - length
    ):

        result = evaluate_outage(
            start,
            length
        )

        if result is not None:

            trials.append(result)

    errors = np.array([
        r["error"]
        for r in trials
    ])

    print(
        f"\n===== {name} ====="
    )

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

    for r in trials:

        r["outage_class"] = name

        results.append(r)


# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== SAVED =====")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STEP 25 COMPLETE")
print("=" * 70)