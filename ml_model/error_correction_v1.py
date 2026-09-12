import pandas as pd
import numpy as np
import joblib


print("=" * 75)
print("PROJECT PRIME — ERROR CORRECTION V1")
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
    r"\PROJECT_PRIME\data\error_correction_v1_results.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

imputer = joblib.load(IMPUTER_PATH)


print("\n===== DATA =====")

print("Samples:", len(df))


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


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)

val_end = int(n * 0.85)


train_df = df.iloc[
    :train_end
].reset_index(drop=True)


test_df = df.iloc[
    val_end:
].reset_index(drop=True)


print("\n===== SPLIT =====")

print("Train:", len(train_df))

print("Test:", len(test_df))


# ============================================================
# PREPARE TRAIN DATA
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
]

X_train = imputer.transform(
    X_train
)


train_pred = model.predict(
    X_train
)


train_true_n = train_df[
    "target_north"
].values

train_true_e = train_df[
    "target_east"
].values


# ============================================================
# CALCULATE SYSTEMATIC BIAS
# ============================================================

train_error_n = (
    train_true_n
    -
    train_pred[:, 0]
)

train_error_e = (
    train_true_e
    -
    train_pred[:, 1]
)


bias_n = np.mean(
    train_error_n
)

bias_e = np.mean(
    train_error_e
)


print("\n===== LEARNED BIAS =====")

print(
    "North bias:",
    round(bias_n, 4),
    "m"
)

print(
    "East bias:",
    round(bias_e, 4),
    "m"
)


# ============================================================
# TEST DATA
# ============================================================

X_test = test_df[
    FEATURE_COLUMNS
]

X_test = imputer.transform(
    X_test
)


test_pred = model.predict(
    X_test
)


true_n = test_df[
    "target_north"
].values

true_e = test_df[
    "target_east"
].values


# ============================================================
# BASE RF
# ============================================================

base_pred_n = test_pred[:, 0]

base_pred_e = test_pred[:, 1]


base_errors = np.sqrt(

    (
        base_pred_n
        -
        true_n
    ) ** 2

    +

    (
        base_pred_e
        -
        true_e
    ) ** 2
)


# ============================================================
# CORRECTED RF
# ============================================================

corrected_pred_n = (
    base_pred_n
    +
    bias_n
)

corrected_pred_e = (
    base_pred_e
    +
    bias_e
)


corrected_errors = np.sqrt(

    (
        corrected_pred_n
        -
        true_n
    ) ** 2

    +

    (
        corrected_pred_e
        -
        true_e
    ) ** 2
)


# ============================================================
# FUNCTION
# ============================================================

def print_stats(
    name,
    errors
):

    print(
        f"\n{name}"
    )

    print(
        "Mean:",
        round(
            np.mean(errors),
            3
        ),
        "m"
    )

    print(
        "Median:",
        round(
            np.median(errors),
            3
        ),
        "m"
    )

    print(
        "P90:",
        round(
            np.percentile(
                errors,
                90
            ),
            3
        ),
        "m"
    )

    print(
        "Maximum:",
        round(
            np.max(errors),
            3
        ),
        "m"
    )


# ============================================================
# RESULTS
# ============================================================

print("\n===== TEST RESULTS =====")

print_stats(
    "BASE RF",
    base_errors
)

print_stats(
    "BIAS CORRECTED RF",
    corrected_errors
)


# ============================================================
# IMPROVEMENT
# ============================================================

base_mean = np.mean(
    base_errors
)

corrected_mean = np.mean(
    corrected_errors
)


improvement = (
    (
        base_mean
        -
        corrected_mean
    )
    /
    base_mean
) * 100


print(
    "\nMean error improvement:",
    round(
        improvement,
        2
    ),
    "%"
)


# ============================================================
# STATEFUL OUTAGE TEST
# ============================================================

OUTAGE_DURATIONS = [
    10,
    20,
    30,
    60
]


results = []


for duration in OUTAGE_DURATIONS:

    errors_base = []

    errors_corrected = []


    for start in range(
        len(test_df)
    ):

        elapsed = 0.0

        end = start


        true_n_total = 0.0
        true_e_total = 0.0

        pred_n_total = 0.0
        pred_e_total = 0.0

        corr_n_total = 0.0
        corr_e_total = 0.0


        while end < len(test_df):

            dt = test_df[
                "fix_interval_seconds"
            ].iloc[end]


            if (
                not np.isfinite(dt)
                or dt <= 0
            ):
                break


            if (
                elapsed + dt > duration
                and end > start
            ):
                break


            elapsed += dt


            true_n_total += true_n[end]

            true_e_total += true_e[end]


            pred_n_total += base_pred_n[end]

            pred_e_total += base_pred_e[end]


            corr_n_total += corrected_pred_n[end]

            corr_e_total += corrected_pred_e[end]


            end += 1


        if end <= start:

            continue


        true_vector = np.array([
            true_n_total,
            true_e_total
        ])


        base_vector = np.array([
            pred_n_total,
            pred_e_total
        ])


        corrected_vector = np.array([
            corr_n_total,
            corr_e_total
        ])


        base_error = np.linalg.norm(
            base_vector
            -
            true_vector
        )


        corrected_error = np.linalg.norm(
            corrected_vector
            -
            true_vector
        )


        errors_base.append(
            base_error
        )

        errors_corrected.append(
            corrected_error
        )


    errors_base = np.array(
        errors_base
    )

    errors_corrected = np.array(
        errors_corrected
    )


    print(
        "\n---------------------------------------------"
    )

    print(
        f"OUTAGE: {duration} seconds"
    )

    print(
        "---------------------------------------------"
    )


    print(
        "Base RF mean:",
        round(
            np.mean(errors_base),
            3
        ),
        "m"
    )


    print(
        "Corrected RF mean:",
        round(
            np.mean(errors_corrected),
            3
        ),
        "m"
    )


    improvement = (

        (
            np.mean(errors_base)
            -
            np.mean(errors_corrected)
        )
        /
        np.mean(errors_base)
    ) * 100


    print(
        "Improvement:",
        round(
            improvement,
            2
        ),
        "%"
    )


    results.append({

        "outage_seconds":
            duration,

        "base_mean_error_m":
            np.mean(
                errors_base
            ),

        "corrected_mean_error_m":
            np.mean(
                errors_corrected
            ),

        "improvement_percent":
            improvement
    })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
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
    "STEP 29 COMPLETE"
)

print(
    "=" * 75
)