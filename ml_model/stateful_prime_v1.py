import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("PROJECT PRIME — STATEFUL TRAJECTORY SIMULATOR V1")
print("=" * 70)

# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_random_forest_v2.pkl"

IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_imputer_v2.pkl"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\stateful_prime_v1_results.csv"


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

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = df[feature_columns]

X = imputer.transform(X)

# Predict all motion increments
pred = model.predict(X)

pred_north = pred[:, 0]
pred_east = pred[:, 1]

true_north = df["target_north"].values
true_east = df["target_east"].values

intervals = df["fix_interval_seconds"].values


# =========================================================
# STATEFUL SIMULATION
# =========================================================

def simulate_outage(outage_seconds):

    # Estimated state
    est_north = 0.0
    est_east = 0.0

    # Actual state
    true_north = 0.0
    true_east = 0.0

    elapsed = 0.0

    trajectory = []

    for i in range(len(df)):

        interval = intervals[i]

        # Don't exceed requested outage
        if elapsed + interval > outage_seconds:
            break

        # -------------------------------------------------
        # ML STATE UPDATE
        # -------------------------------------------------

        est_north += pred_north[i]
        est_east += pred_east[i]

        # -------------------------------------------------
        # TRUE STATE
        # -------------------------------------------------

        true_north += df["target_north"].iloc[i]
        true_east += df["target_east"].iloc[i]

        elapsed += interval

        # -------------------------------------------------
        # POSITION ERROR
        # -------------------------------------------------

        position_error = np.sqrt(
            (est_north - true_north) ** 2 +
            (est_east - true_east) ** 2
        )

        trajectory.append({
            "interval_index": i,
            "elapsed_time": elapsed,
            "estimated_north": est_north,
            "estimated_east": est_east,
            "true_north": true_north,
            "true_east": true_east,
            "position_error": position_error
        })

    trajectory_df = pd.DataFrame(trajectory)

    if len(trajectory_df) == 0:
        return None

    return trajectory_df


# =========================================================
# TEST OUTAGES
# =========================================================

outages = [10, 20, 30, 60]

all_results = []

print("\n===== STATEFUL OUTAGE TESTS =====")

for outage in outages:

    traj = simulate_outage(outage)

    if traj is None:
        continue

    final_error = traj["position_error"].iloc[-1]

    mean_error = traj["position_error"].mean()

    max_error = traj["position_error"].max()

    final_true_distance = np.sqrt(
        traj["true_north"].iloc[-1] ** 2 +
        traj["true_east"].iloc[-1] ** 2
    )

    final_est_distance = np.sqrt(
        traj["estimated_north"].iloc[-1] ** 2 +
        traj["estimated_east"].iloc[-1] ** 2
    )

    print(f"\n===== {outage} SECOND OUTAGE =====")

    print(
        "Actual duration:",
        round(traj["elapsed_time"].iloc[-1], 2),
        "s"
    )

    print(
        "Intervals:",
        len(traj)
    )

    print(
        "True displacement:",
        round(final_true_distance, 2),
        "m"
    )

    print(
        "Estimated displacement:",
        round(final_est_distance, 2),
        "m"
    )

    print(
        "Mean trajectory error:",
        round(mean_error, 2),
        "m"
    )

    print(
        "Final position error:",
        round(final_error, 2),
        "m"
    )

    print(
        "Maximum position error:",
        round(max_error, 2),
        "m"
    )

    # Save summary
    all_results.append({
        "requested_outage": outage,
        "actual_outage": traj["elapsed_time"].iloc[-1],
        "intervals": len(traj),
        "true_displacement": final_true_distance,
        "estimated_displacement": final_est_distance,
        "mean_error": mean_error,
        "final_error": final_error,
        "max_error": max_error
    })


# =========================================================
# SAVE SUMMARY
# =========================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== RESULTS SAVED =====")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STEP 20 COMPLETE")
print("=" * 70)