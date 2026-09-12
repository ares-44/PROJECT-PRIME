import os
import joblib
import numpy as np
import pandas as pd

print("=" * 75)
print("PROJECT PRIME — CONFIDENCE-AWARE TRAJECTORY V1")
print("=" * 75)

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(
    BASE,
    "data",
    "prime_ml_dataset_v3.csv"
)

MODEL = os.path.join(
    BASE,
    "models",
    "prime_random_forest_v2.pkl"
)

IMPUTER = os.path.join(
    BASE,
    "models",
    "prime_imputer_v2.pkl"
)

OUT = os.path.join(
    BASE,
    "data",
    "confidence_trajectory_v1_results.csv"
)

STATEFUL_OUT = os.path.join(
    BASE,
    "data",
    "confidence_trajectory_v1_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

print("\n===== LOADING MODEL =====")

df = pd.read_csv(DATA)

model = joblib.load(MODEL)
imputer = joblib.load(IMPUTER)

print("Model loaded successfully.")
print("Imputer loaded successfully.")


# ============================================================
# GET EXACT FEATURES USED DURING TRAINING
# ============================================================

print("\n===== FEATURES =====")

if hasattr(imputer, "feature_names_in_"):

    feature_cols = list(
        imputer.feature_names_in_
    )

    print(
        "Using feature names stored in imputer."
    )

else:

    print(
        "WARNING: Imputer has no feature_names_in_."
    )

    # Fallback: use the 48 feature columns
    # excluding targets and metadata.

    excluded = {
        "target_north",
        "target_east",
        "target_distance",
        "fix_interval_seconds",
        "gps_fix_index"
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    print(
        "Using fallback feature selection."
    )


print(
    "Number of features:",
    len(feature_cols)
)

print("\nFeature list:")

for i, feature in enumerate(
    feature_cols,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    c for c in feature_cols
    if c not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(
            " -",
            feature
        )

    raise ValueError(
        "Dataset does not contain all features "
        "required by the trained imputer."
    )


# ============================================================
# PREPARE X / Y
# ============================================================

X_df = df[feature_cols].copy()

y = df[
    [
        "target_north",
        "target_east"
    ]
].values


# Keep DataFrame so sklearn sees the same
# feature names used during training.

X = imputer.transform(X_df)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)


X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[
    train_end:val_end
]

y_val = y[
    train_end:val_end
]

X_test = X[
    val_end:
]

y_test = y[
    val_end:
]


print("\n===== DATA =====")

print(
    "Total samples :",
    n
)

print(
    "Features      :",
    X.shape[1]
)

print(
    "Train samples :",
    len(X_train)
)

print(
    "Validation    :",
    len(X_val)
)

print(
    "Test samples  :",
    len(X_test)
)


# ============================================================
# TREE PREDICTIONS
# ============================================================

def get_predictions(X_data):

    tree_predictions = np.stack(
        [
            tree.predict(X_data)
            for tree in model.estimators_
        ],
        axis=0
    )

    mean_prediction = (
        tree_predictions.mean(
            axis=0
        )
    )

    std_prediction = (
        tree_predictions.std(
            axis=0
        )
    )

    uncertainty = np.sqrt(
        std_prediction[:, 0] ** 2
        +
        std_prediction[:, 1] ** 2
    )

    return (
        mean_prediction,
        uncertainty,
        std_prediction
    )


train_pred, train_unc, train_std = (
    get_predictions(X_train)
)

val_pred, val_unc, val_std = (
    get_predictions(X_val)
)

test_pred, test_unc, test_std = (
    get_predictions(X_test)
)


print("\n===== TREE PREDICTIONS =====")

print(
    "Trees:",
    len(model.estimators_)
)

print(
    "Prediction shape:",
    test_pred.shape
)


# ============================================================
# MOTION PRIOR
# ============================================================

motion_prior = np.mean(
    y_train,
    axis=0
)

print("\n===== MOTION PRIOR =====")

print(
    "North:",
    round(
        motion_prior[0],
        6
    ),
    "m"
)

print(
    "East :",
    round(
        motion_prior[1],
        6
    ),
    "m"
)


# ============================================================
# UNCERTAINTY SCALE
# ============================================================

uncertainty_scale = np.median(
    train_unc
)

print(
    "\nTraining uncertainty scale:",
    round(
        uncertainty_scale,
        4
    ),
    "m"
)


# ============================================================
# CONFIDENCE FUNCTION
# ============================================================

def apply_confidence(
    predictions,
    uncertainty,
    strength
):

    confidence = 1.0 / (
        1.0
        +
        strength
        *
        uncertainty
        /
        (
            uncertainty_scale
            +
            1e-8
        )
    )

    corrected = (
        confidence[:, None]
        *
        predictions
        +
        (
            1.0
            -
            confidence[:, None]
        )
        *
        motion_prior
    )

    return (
        corrected,
        confidence
    )


# ============================================================
# VALIDATION BASELINE
# ============================================================

base_val_error = np.linalg.norm(
    val_pred - y_val,
    axis=1
)

print(
    "\n===== VALIDATION BASELINE ====="
)

print(
    "Mean   :",
    round(
        np.mean(base_val_error),
        4
    ),
    "m"
)

print(
    "Median :",
    round(
        np.median(base_val_error),
        4
    ),
    "m"
)

print(
    "P90    :",
    round(
        np.percentile(
            base_val_error,
            90
        ),
        4
    ),
    "m"
)


# ============================================================
# TUNE STRENGTH
# ============================================================

strengths = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0
]

best_strength = None

best_error = float(
    "inf"
)


print(
    "\n===== VALIDATION TUNING ====="
)


for strength in strengths:

    corrected, confidence = (
        apply_confidence(
            val_pred,
            val_unc,
            strength
        )
    )

    errors = np.linalg.norm(
        corrected - y_val,
        axis=1
    )

    mean_error = np.mean(
        errors
    )

    print(
        f"Strength {strength:>4}: "
        f"Mean Error = "
        f"{mean_error:.3f} m"
    )

    if mean_error < best_error:

        best_error = mean_error

        best_strength = strength


print(
    "\n===== BEST PARAMETER ====="
)

print(
    "Best strength:",
    best_strength
)

print(
    "Best validation error:",
    round(
        best_error,
        4
    ),
    "m"
)


# ============================================================
# TEST
# ============================================================

base_test_error = np.linalg.norm(
    test_pred - y_test,
    axis=1
)


corrected_test, test_confidence = (
    apply_confidence(
        test_pred,
        test_unc,
        best_strength
    )
)


corrected_test_error = np.linalg.norm(
    corrected_test - y_test,
    axis=1
)


# ============================================================
# STATISTICS
# ============================================================

def stats(errors):

    return {
        "mean":
            float(
                np.mean(errors)
            ),

        "median":
            float(
                np.median(errors)
            ),

        "p90":
            float(
                np.percentile(
                    errors,
                    90
                )
            ),

        "max":
            float(
                np.max(errors)
            )
    }


base = stats(
    base_test_error
)

corrected = stats(
    corrected_test_error
)


improvement = (
    (
        base["mean"]
        -
        corrected["mean"]
    )
    /
    base["mean"]
) * 100


# ============================================================
# TEST RESULTS
# ============================================================

print(
    "\n===== TEST RESULTS ====="
)

print(
    "\nBASE RF"
)

print(
    "Mean   :",
    round(
        base["mean"],
        4
    ),
    "m"
)

print(
    "Median :",
    round(
        base["median"],
        4
    ),
    "m"
)

print(
    "P90    :",
    round(
        base["p90"],
        4
    ),
    "m"
)

print(
    "Maximum:",
    round(
        base["max"],
        4
    ),
    "m"
)


print(
    "\nCONFIDENCE-AWARE RF"
)

print(
    "Mean   :",
    round(
        corrected["mean"],
        4
    ),
    "m"
)

print(
    "Median :",
    round(
        corrected["median"],
        4
    ),
    "m"
)

print(
    "P90    :",
    round(
        corrected["p90"],
        4
    ),
    "m"
)

print(
    "Maximum:",
    round(
        corrected["max"],
        4
    ),
    "m"
)


print(
    "\n===== IMPROVEMENT ====="
)

print(
    "Mean error improvement:",
    round(
        improvement,
        2
    ),
    "%"
)


# ============================================================
# STATEFUL OUTAGE TEST
# ============================================================

print(
    "\n===== STATEFUL OUTAGE TEST ====="
)

durations = [
    10,
    20,
    30,
    60
]

rows = []


interval_times = (
    df[
        "fix_interval_seconds"
    ]
    .values[
        val_end:
    ]
)


for duration in durations:

    horizon_results = []


    for start in range(
        len(test_pred)
    ):

        elapsed = 0.0

        true_pos = np.array(
            [0.0, 0.0]
        )

        base_pos = np.array(
            [0.0, 0.0]
        )

        conf_pos = np.array(
            [0.0, 0.0]
        )


        i = start


        while (
            i < len(test_pred)
            and elapsed < duration
        ):

            dt = (
                interval_times[i]
            )

            true_pos += (
                y_test[i]
            )

            base_pos += (
                test_pred[i]
            )

            conf_pos += (
                corrected_test[i]
            )

            elapsed += dt

            i += 1


        if i > start:

            base_error = np.linalg.norm(
                base_pos - true_pos
            )

            conf_error = np.linalg.norm(
                conf_pos - true_pos
            )

            horizon_results.append(
                (
                    elapsed,
                    base_error,
                    conf_error
                )
            )


    if horizon_results:

        arr = np.array(
            horizon_results
        )

        base_errors = arr[:, 1]

        conf_errors = arr[:, 2]


        print(
            f"\n{duration} SEC"
        )

        print(
            "Trials:",
            len(base_errors)
        )

        print(
            "Base mean:",
            round(
                np.mean(
                    base_errors
                ),
                3
            ),
            "m"
        )

        print(
            "Confidence mean:",
            round(
                np.mean(
                    conf_errors
                ),
                3
            ),
            "m"
        )

        print(
            "Base median:",
            round(
                np.median(
                    base_errors
                ),
                3
            ),
            "m"
        )

        print(
            "Confidence median:",
            round(
                np.median(
                    conf_errors
                ),
                3
            ),
            "m"
        )

        print(
            "Base P90:",
            round(
                np.percentile(
                    base_errors,
                    90
                ),
                3
            ),
            "m"
        )

        print(
            "Confidence P90:",
            round(
                np.percentile(
                    conf_errors,
                    90
                ),
                3
            ),
            "m"
        )


        rows.append(
            {
                "duration_requested_sec":
                    duration,

                "trials":
                    len(base_errors),

                "base_mean_error_m":
                    np.mean(
                        base_errors
                    ),

                "confidence_mean_error_m":
                    np.mean(
                        conf_errors
                    ),

                "base_median_error_m":
                    np.median(
                        base_errors
                    ),

                "confidence_median_error_m":
                    np.median(
                        conf_errors
                    ),

                "base_p90_error_m":
                    np.percentile(
                        base_errors,
                        90
                    ),

                "confidence_p90_error_m":
                    np.percentile(
                        conf_errors,
                        90
                    )
            }
        )


# ============================================================
# SAVE SAMPLE RESULTS
# ============================================================

sample_results = pd.DataFrame(
    {
        "actual_north_m":
            y_test[:, 0],

        "actual_east_m":
            y_test[:, 1],

        "predicted_north_m":
            test_pred[:, 0],

        "predicted_east_m":
            test_pred[:, 1],

        "corrected_north_m":
            corrected_test[:, 0],

        "corrected_east_m":
            corrected_test[:, 1],

        "uncertainty_m":
            test_unc,

        "confidence":
            test_confidence,

        "base_error_m":
            base_test_error,

        "confidence_error_m":
            corrected_test_error
    }
)


sample_results.to_csv(
    OUT,
    index=False
)


# ============================================================
# SAVE STATEFUL SUMMARY
# ============================================================

result_df = pd.DataFrame(
    rows
)

result_df.to_csv(
    STATEFUL_OUT,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print(
    "\n===== SAVED ====="
)

print(
    "Detailed results:"
)

print(
    OUT
)

print(
    "\nStateful summary:"
)

print(
    STATEFUL_OUT
)

print(
    "\n" + "=" * 75
)

print(
    "STEP 31 COMPLETE"
)

print(
    "=" * 75
)