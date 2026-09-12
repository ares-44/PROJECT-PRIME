import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("PROJECT PRIME — GNSS OUTAGE SIMULATION V1")
print("=" * 70)

# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_random_forest_v2.pkl"

IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_imputer_v2.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))

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

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = df[feature_columns]

X = imputer.transform(X)

# =========================================================
# PREDICTIONS
# =========================================================

pred = model.predict(X)

pred_north = pred[:, 0]
pred_east = pred[:, 1]

true_north = df["target_north"].values
true_east = df["target_east"].values

intervals = df["fix_interval_seconds"].values

# =========================================================
# FUNCTION: SIMULATE OUTAGE
# =========================================================

def simulate_outage(outage_seconds):

    cumulative_pred_n = 0.0
    cumulative_pred_e = 0.0

    cumulative_true_n = 0.0
    cumulative_true_e = 0.0

    elapsed = 0.0
    used_intervals = 0

    for i in range(len(df)):

        interval = intervals[i]

        if elapsed + interval > outage_seconds:
            break

        cumulative_pred_n += pred_north[i]
        cumulative_pred_e += pred_east[i]

        cumulative_true_n += true_north[i]
        cumulative_true_e += true_east[i]

        elapsed += interval
        used_intervals += 1

    final_error = np.sqrt(
        (cumulative_pred_n - cumulative_true_n) ** 2
        +
        (cumulative_pred_e - cumulative_true_e) ** 2
    )

    true_distance = np.sqrt(
        cumulative_true_n ** 2 +
        cumulative_true_e ** 2
    )

    predicted_distance = np.sqrt(
        cumulative_pred_n ** 2 +
        cumulative_pred_e ** 2
    )

    return {
        "requested_outage": outage_seconds,
        "actual_outage": elapsed,
        "intervals": used_intervals,
        "true_distance": true_distance,
        "predicted_distance": predicted_distance,
        "final_error": final_error
    }


# =========================================================
# RUN TESTS
# =========================================================

outages = [10, 20, 30, 60]

results = []

print("\n===== OUTAGE TESTS =====")

for seconds in outages:

    result = simulate_outage(seconds)

    results.append(result)

    print(
        f"\n{seconds} SECOND OUTAGE"
    )

    print(
        "Actual duration:",
        round(result["actual_outage"], 2),
        "s"
    )

    print(
        "Intervals:",
        result["intervals"]
    )

    print(
        "True displacement:",
        round(result["true_distance"], 2),
        "m"
    )

    print(
        "Predicted displacement:",
        round(result["predicted_distance"], 2),
        "m"
    )

    print(
        "FINAL POSITION ERROR:",
        round(result["final_error"], 2),
        "m"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\gnss_outage_simulation_v1.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== SAVED =====")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STEP 18 COMPLETE")
print("=" * 70)