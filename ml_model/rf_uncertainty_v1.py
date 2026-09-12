import pandas as pd
import numpy as np
import joblib

from scipy.stats import spearmanr


print("=" * 75)
print("PROJECT PRIME — RF UNCERTAINTY ANALYSIS V1")
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
    r"\PROJECT_PRIME\data\rf_uncertainty_v1_results.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

model = joblib.load(
    MODEL_PATH
)

imputer = joblib.load(
    IMPUTER_PATH
)


print("\n===== DATA =====")

print(
    "Samples:",
    len(df)
)


# ============================================================
# FEATURES
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
# CHRONOLOGICAL TEST SET
# ============================================================

n = len(df)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)


test_df = df.iloc[
    val_end:
].reset_index(
    drop=True
)


print("\n===== TEST SET =====")

print(
    "Test samples:",
    len(test_df)
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X_test = test_df[
    FEATURE_COLUMNS
]

X_test = imputer.transform(
    X_test
)


# ============================================================
# TRUE TARGET
# ============================================================

true_n = test_df[
    "target_north"
].values

true_e = test_df[
    "target_east"
].values


# ============================================================
# INDIVIDUAL TREE PREDICTIONS
# ============================================================

print("\n===== TREE PREDICTIONS =====")

tree_predictions = []


for tree in model.estimators_:

    prediction = tree.predict(
        X_test
    )

    tree_predictions.append(
        prediction
    )


tree_predictions = np.array(
    tree_predictions
)


print(
    "Trees:",
    tree_predictions.shape[0]
)

print(
    "Prediction shape:",
    tree_predictions.shape
)


# Expected:

# (number_of_trees, test_samples, 2)


# ============================================================
# MEAN PREDICTION
# ============================================================

pred_n = np.mean(
    tree_predictions[:, :, 0],
    axis=0
)

pred_e = np.mean(
    tree_predictions[:, :, 1],
    axis=0
)


# ============================================================
# UNCERTAINTY
# ============================================================

std_n = np.std(
    tree_predictions[:, :, 0],
    axis=0
)

std_e = np.std(
    tree_predictions[:, :, 1],
    axis=0
)


# Combined directional uncertainty

uncertainty = np.sqrt(
    std_n ** 2
    +
    std_e ** 2
)


# ============================================================
# ACTUAL PREDICTION ERROR
# ============================================================

actual_error = np.sqrt(

    (
        pred_n
        -
        true_n
    ) ** 2

    +

    (
        pred_e
        -
        true_e
    ) ** 2
)


# ============================================================
# STATISTICS
# ============================================================

print(
    "\n===== UNCERTAINTY ====="
)

print(
    "Mean:",
    round(
        np.mean(uncertainty),
        4
    ),
    "m"
)

print(
    "Median:",
    round(
        np.median(uncertainty),
        4
    ),
    "m"
)

print(
    "P90:",
    round(
        np.percentile(
            uncertainty,
            90
        ),
        4
    ),
    "m"
)

print(
    "Maximum:",
    round(
        np.max(uncertainty),
        4
    ),
    "m"
)


# ============================================================
# ERROR STATISTICS
# ============================================================

print(
    "\n===== ACTUAL ERROR ====="
)

print(
    "Mean:",
    round(
        np.mean(actual_error),
        4
    ),
    "m"
)

print(
    "Median:",
    round(
        np.median(actual_error),
        4
    ),
    "m"
)

print(
    "P90:",
    round(
        np.percentile(
            actual_error,
            90
        ),
        4
    ),
    "m"
)

print(
    "Maximum:",
    round(
        np.max(actual_error),
        4
    ),
    "m"
)


# ============================================================
# CORRELATION
# ============================================================

spearman_corr, p_value = spearmanr(
    uncertainty,
    actual_error
)


print(
    "\n===== UNCERTAINTY vs ERROR ====="
)

print(
    "Spearman correlation:",
    round(
        spearman_corr,
        4
    )
)

print(
    "P-value:",
    round(
        p_value,
        6
    )
)


# ============================================================
# CONFIDENCE GROUPS
# ============================================================

print(
    "\n===== UNCERTAINTY GROUP ANALYSIS ====="
)


q25 = np.percentile(
    uncertainty,
    25
)

q50 = np.percentile(
    uncertainty,
    50
)

q75 = np.percentile(
    uncertainty,
    75
)


groups = [

    (
        "LOW",
        uncertainty <= q25
    ),

    (
        "MEDIUM-LOW",
        (
            uncertainty > q25
        )
        &
        (
            uncertainty <= q50
        )
    ),

    (
        "MEDIUM-HIGH",
        (
            uncertainty > q50
        )
        &
        (
            uncertainty <= q75
        )
    ),

    (
        "HIGH",
        uncertainty > q75
    )
]


for name, mask in groups:

    if np.sum(mask) == 0:

        continue


    print(
        f"\n{name}"
    )

    print(
        "Samples:",
        np.sum(mask)
    )

    print(
        "Mean uncertainty:",
        round(
            np.mean(
                uncertainty[mask]
            ),
            3
        ),
        "m"
    )

    print(
        "Mean actual error:",
        round(
            np.mean(
                actual_error[mask]
            ),
            3
        ),
        "m"
    )

    print(
        "P90 actual error:",
        round(
            np.percentile(
                actual_error[mask],
                90
            ),
            3
        ),
        "m"
    )


# ============================================================
# HIGH UNCERTAINTY ERROR RATE
# ============================================================

high_threshold = q75

high_mask = (
    uncertainty >= high_threshold
)


print(
    "\n===== HIGH UNCERTAINTY ====="
)

print(
    "Threshold:",
    round(
        high_threshold,
        4
    ),
    "m"
)

print(
    "High uncertainty samples:",
    np.sum(high_mask)
)

print(
    "Mean error:",
    round(
        np.mean(
            actual_error[high_mask]
        ),
        3
    ),
    "m"
)


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame({

    "true_north":
        true_n,

    "true_east":
        true_e,

    "pred_north":
        pred_n,

    "pred_east":
        pred_e,

    "uncertainty_north":
        std_n,

    "uncertainty_east":
        std_e,

    "uncertainty_total":
        uncertainty,

    "actual_error":
        actual_error
})


results.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\n===== SAVED ====="
)

print(
    OUTPUT_PATH
)


print(
    "\n" + "=" * 75
)

print(
    "STEP 30 COMPLETE"
)

print(
    "=" * 75
)