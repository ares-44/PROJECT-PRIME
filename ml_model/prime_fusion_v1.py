import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("PROJECT PRIME — GNSS + ML SENSOR FUSION V1")
print("=" * 70)

# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_random_forest_v2.pkl"

IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_imputer_v2.pkl"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_fusion_v1_results.csv"


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
# FUSION SIMULATION
# =========================================================

def simulate_fusion(outage_intervals):

    estimated_n = 0.0
    estimated_e = 0.0

    true_n = 0.0
    true_e = 0.0

    errors = []

    outage_count = 0

    for i in range(len(df)):

        # -------------------------------------------------
        # TRUE POSITION
        # -------------------------------------------------

        true_n += true_n[i] if False else 0
        true_e += true_e[i] if False else 0

        # Correct accumulation
        true_position_n = (
            df["target_north"].iloc[:i+1].sum()
        )

        true_position_e = (
            df["target_east"].iloc[:i+1].sum()
        )

        # -------------------------------------------------
        # OUTAGE MODE
        # -------------------------------------------------

        if outage_count < outage_intervals:

            estimated_n += pred_n[i]
            estimated_e += pred_e[i]

            outage_count += 1

        # -------------------------------------------------
        # GNSS AVAILABLE
        # -------------------------------------------------

        else:

            # GNSS correction:
            # directly align estimated state with
            # the true GNSS displacement state

            estimated_n = true_position_n
            estimated_e = true_position_e

            outage_count = 0

        # -------------------------------------------------
        # ERROR
        # -------------------------------------------------

        error = np.sqrt(
            (estimated_n - true_position_n) ** 2 +
            (estimated_e - true_position_e) ** 2
        )

        errors.append(error)

    return np.array(errors)


# =========================================================
# OUTAGE TESTS
# =========================================================

tests = {
    "10_sec": 1,
    "20_sec": 2,
    "30_sec": 3,
    "60_sec": 6
}

results = []

print("\n===== FUSION TESTS =====")

for name, outage in tests.items():

    errors = simulate_fusion(outage)

    print(f"\n{name}")

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

    results.append({
        "outage": name,
        "mean_error": np.mean(errors),
        "median_error": np.median(errors),
        "p90_error": np.percentile(errors, 90),
        "max_error": np.max(errors)
    })


# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== SAVED =====")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STEP 21 COMPLETE")
print("=" * 70)