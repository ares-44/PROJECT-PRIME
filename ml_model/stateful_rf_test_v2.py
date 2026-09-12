import pandas as pd
import numpy as np
import joblib


print("=" * 75)
print("PROJECT PRIME — STATEFUL RF OUTAGE EVALUATION V2")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop"
    r"\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"
)

MODEL_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop"
    r"\PROJECT_PRIME\models\prime_random_forest_v2.pkl"
)

IMPUTER_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop"
    r"\PROJECT_PRIME\models\prime_imputer_v2.pkl"
)

OUTPUT_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop"
    r"\PROJECT_PRIME\data\stateful_rf_test_v2_results.csv"
)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

imputer = joblib.load(IMPUTER_PATH)


print("\n===== DATA =====")

print("Total samples:", len(df))

print("Columns:", len(df.columns))


# ============================================================
# TARGET / METADATA
# ============================================================

TARGET_COLUMNS = [
    "target_north",
    "target_east",
    "target_distance"
]

METADATA_COLUMNS = [
    "fix_interval_seconds",
    "gps_fix_index"
]


FEATURE_COLUMNS = [
    c for c in df.columns
    if c not in TARGET_COLUMNS + METADATA_COLUMNS
]


print(
    "Features:",
    len(FEATURE_COLUMNS)
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)

val_end = int(n * 0.85)

test_start = val_end


print("\n===== CHRONOLOGICAL SPLIT =====")

print("Train:", train_end)

print(
    "Validation:",
    val_end - train_end
)

print(
    "Test:",
    n - val_end
)

print(
    "Test starts at:",
    test_start
)


# ============================================================
# TEST DATA ONLY
# ============================================================

test_df = df.iloc[
    test_start:
].reset_index(
    drop=True
)


X_test = test_df[
    FEATURE_COLUMNS
]


X_test = imputer.transform(
    X_test
)


# ============================================================
# PREDICT EACH INTERVAL
# ============================================================

pred = model.predict(
    X_test
)


pred_north = pred[:, 0]

pred_east = pred[:, 1]


true_north = test_df[
    "target_north"
].values

true_east = test_df[
    "target_east"
].values


intervals = test_df[
    "fix_interval_seconds"
].values


print(
    "\n===== INTERVAL PREDICTIONS ====="
)

print(
    "Predictions generated:",
    len(pred)
)


# ============================================================
# OUTAGE DURATIONS
# ============================================================

OUTAGE_DURATIONS = [
    10,
    20,
    30,
    60
]


results = []


# ============================================================
# OUTAGE EVALUATION
# ============================================================

for requested_duration in OUTAGE_DURATIONS:

    print("\n" + "-" * 75)

    print(
        f"OUTAGE: {requested_duration} seconds"
    )

    print("-" * 75)


    errors = []

    true_distances = []

    estimated_distances = []

    actual_durations = []

    interval_counts = []


    # --------------------------------------------------------
    # Start outage at every point in TEST SET
    # --------------------------------------------------------

    for start in range(
        len(test_df)
    ):

        elapsed = 0.0

        end = start


        cumulative_true_n = 0.0
        cumulative_true_e = 0.0

        cumulative_pred_n = 0.0
        cumulative_pred_e = 0.0


        # ----------------------------------------------------
        # Accumulate intervals WITHOUT crossing target duration
        # ----------------------------------------------------

        while end < len(test_df):

            dt = intervals[end]


            # Safety check
            if (
                not np.isfinite(dt)
                or dt <= 0
            ):
                break


            # Do not cross requested outage duration.
            #
            # Example:
            # 10 sec outage
            # interval ≈ 9 sec
            # → use 1 interval
            #
            # 20 sec outage
            # intervals ≈ 9 + 9
            # → use 2 intervals
            #
            if (
                elapsed + dt > requested_duration
                and end > start
            ):
                break


            # Add interval duration
            elapsed += dt


            # Add TRUE displacement
            cumulative_true_n += (
                true_north[end]
            )

            cumulative_true_e += (
                true_east[end]
            )


            # Add PREDICTED displacement
            cumulative_pred_n += (
                pred_north[end]
            )

            cumulative_pred_e += (
                pred_east[end]
            )


            # IMPORTANT:
            # Move to next interval
            end += 1


        # ----------------------------------------------------
        # Need at least one interval
        # ----------------------------------------------------

        if end <= start:

            continue


        # ====================================================
        # TRUE / PREDICTED VECTORS
        # ====================================================

        true_vector = np.array([
            cumulative_true_n,
            cumulative_true_e
        ])


        predicted_vector = np.array([
            cumulative_pred_n,
            cumulative_pred_e
        ])


        # ====================================================
        # POSITION ERROR
        # ====================================================

        error = np.linalg.norm(
            predicted_vector
            -
            true_vector
        )


        # ====================================================
        # DISTANCES
        # ====================================================

        true_distance = np.linalg.norm(
            true_vector
        )


        estimated_distance = np.linalg.norm(
            predicted_vector
        )


        # ====================================================
        # STORE
        # ====================================================

        errors.append(
            error
        )

        true_distances.append(
            true_distance
        )

        estimated_distances.append(
            estimated_distance
        )

        actual_durations.append(
            elapsed
        )

        interval_counts.append(
            end - start
        )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    errors = np.array(
        errors
    )

    true_distances = np.array(
        true_distances
    )

    estimated_distances = np.array(
        estimated_distances
    )

    actual_durations = np.array(
        actual_durations
    )

    interval_counts = np.array(
        interval_counts
    )


    # ========================================================
    # NO VALID TRIALS
    # ========================================================

    if len(errors) == 0:

        print(
            "No valid trials."
        )

        continue


    # ========================================================
    # STATISTICS
    # ========================================================

    mean_error = np.mean(
        errors
    )

    median_error = np.median(
        errors
    )

    p90_error = np.percentile(
        errors,
        90
    )

    max_error = np.max(
        errors
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "Trials:",
        len(errors)
    )


    print(
        "Mean actual duration:",
        round(
            np.mean(
                actual_durations
            ),
            2
        ),
        "sec"
    )


    print(
        "Mean intervals:",
        round(
            np.mean(
                interval_counts
            ),
            2
        )
    )


    print(
        "Mean error:",
        round(
            mean_error,
            3
        ),
        "m"
    )


    print(
        "Median error:",
        round(
            median_error,
            3
        ),
        "m"
    )


    print(
        "P90 error:",
        round(
            p90_error,
            3
        ),
        "m"
    )


    print(
        "Maximum error:",
        round(
            max_error,
            3
        ),
        "m"
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "requested_duration_sec":
            requested_duration,

        "trials":
            len(errors),

        "mean_actual_duration_sec":
            np.mean(
                actual_durations
            ),

        "mean_intervals":
            np.mean(
                interval_counts
            ),

        "mean_true_distance_m":
            np.mean(
                true_distances
            ),

        "mean_estimated_distance_m":
            np.mean(
                estimated_distances
            ),

        "mean_error_m":
            mean_error,

        "median_error_m":
            median_error,

        "p90_error_m":
            p90_error,

        "max_error_m":
            max_error
    })


# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n===== RESULTS SAVED ====="
)


print(
    OUTPUT_PATH
)


print(
    "\n" + "=" * 75
)

print(
    "STEP 28 COMPLETE"
)

print(
    "=" * 75
)