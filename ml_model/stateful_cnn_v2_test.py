import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


print("=" * 75)
print("PROJECT PRIME — STATEFUL CNN V2 OUTAGE TEST")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET = os.path.join(
    BASE,
    "data",
    "prime_sequence_dataset_v2.npz"
)

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "prime_temporal_cnn_v2.keras"
)

SCALER_PATH = os.path.join(
    BASE,
    "models",
    "prime_sequence_scaler_v2.pkl"
)

OUT = os.path.join(
    BASE,
    "data",
    "stateful_cnn_v2_results.csv"
)


# ============================================================
# LOAD
# ============================================================

data = np.load(
    DATASET
)

X = data["X"]
y = data["y"]
intervals = data["intervals"]


print("\n===== DATA =====")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    SCALER_PATH,
    "rb"
) as f:

    scaler = pickle.load(f)


# ============================================================
# CHRONOLOGICAL TEST SPLIT
# ============================================================

n = len(X)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)


X_test = X[
    val_end:
]

y_test = y[
    val_end:
]

interval_test = intervals[
    val_end:
]


print("\n===== TEST SET =====")

print(
    "Test samples:",
    len(X_test)
)


# ============================================================
# SCALE
# ============================================================

X_test_flat = X_test.reshape(
    -1,
    X_test.shape[-1]
)

X_test_scaled = scaler.transform(
    X_test_flat
).reshape(
    X_test.shape
)


# ============================================================
# PREDICT ALL TEST SAMPLES
# ============================================================

print(
    "\n===== CNN PREDICTIONS ====="
)

predictions = model.predict(
    X_test_scaled,
    verbose=0
)


print(
    "Prediction shape:",
    predictions.shape
)


# ============================================================
# OUTAGE TEST
# ============================================================

durations = [
    10,
    20,
    30,
    60
]


results = []


print(
    "\n===== STATEFUL OUTAGE RESULTS ====="
)


for duration in durations:

    horizon_results = []


    for start in range(
        len(predictions)
    ):

        elapsed = 0.0

        true_position = np.array(
            [0.0, 0.0]
        )

        predicted_position = np.array(
            [0.0, 0.0]
        )


        i = start


        while (
            i < len(predictions)
            and elapsed < duration
        ):

            dt = interval_test[i]


            # Actual displacement
            true_position += (
                y_test[i]
            )


            # CNN predicted displacement
            predicted_position += (
                predictions[i]
            )


            elapsed += dt

            i += 1


        if i > start:

            error = np.linalg.norm(
                predicted_position
                -
                true_position
            )


            horizon_results.append(
                (
                    elapsed,
                    error
                )
            )


    if horizon_results:

        arr = np.array(
            horizon_results
        )

        errors = arr[:, 1]


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


        print(
            f"\n{duration} SEC"
        )

        print(
            "Trials:",
            len(errors)
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


        results.append(
            {
                "duration_requested_sec":
                    duration,

                "trials":
                    len(errors),

                "mean_error_m":
                    mean_error,

                "median_error_m":
                    median_error,

                "p90_error_m":
                    p90_error,

                "max_error_m":
                    max_error
            }
        )


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUT,
    index=False
)


print(
    "\n===== SAVED ====="
)

print(
    OUT
)


print(
    "\n" + "=" * 75
)

print(
    "STEP 33 COMPLETE"
)

print(
    "=" * 75
)