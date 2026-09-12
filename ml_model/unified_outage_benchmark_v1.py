import os
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SEQ_DATA = os.path.join(
    DATA_DIR,
    "prime_sequence_dataset_v2.npz"
)

HYBRID_DATA = os.path.join(
    DATA_DIR,
    "prime_hybrid_dataset_v1.npz"
)

RF_MODEL = os.path.join(
    MODEL_DIR,
    "prime_random_forest_v2.pkl"
)

RF_IMPUTER = os.path.join(
    MODEL_DIR,
    "prime_imputer_v2.pkl"
)

CNN_V2_MODEL = os.path.join(
    MODEL_DIR,
    "prime_temporal_cnn_v2.keras"
)

CNN_V2_SCALER = os.path.join(
    MODEL_DIR,
    "prime_sequence_scaler_v2.pkl"
)

HYBRID_MODEL = os.path.join(
    MODEL_DIR,
    "prime_hybrid_cnn_v1.keras"
)

HYBRID_SEQ_SCALER = os.path.join(
    MODEL_DIR,
    "prime_hybrid_sequence_scaler_v1.pkl"
)

HYBRID_PHYS_SCALER = os.path.join(
    MODEL_DIR,
    "prime_hybrid_physics_scaler_v1.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

seq_data = np.load(SEQ_DATA)

X_seq = seq_data["X"].astype(np.float32)
y = seq_data["y"].astype(np.float32)

hybrid_data = np.load(HYBRID_DATA)

X_hybrid_seq = hybrid_data["X"].astype(np.float32)
X_physics = hybrid_data["physics_features"].astype(np.float32)

print("=" * 70)
print("DATA")
print("=" * 70)

print("Sequence shape:", X_seq.shape)
print("Target shape:", y.shape)
print("Hybrid sequence:", X_hybrid_seq.shape)
print("Physics features:", X_physics.shape)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

N = len(X_seq)

train_end = int(N * 0.70)
val_end = int(N * 0.85)

test_start = val_end

test_indices = np.arange(
    test_start,
    N
)

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print("Total:", N)
print("Train:", train_end)
print("Validation:", val_end - train_end)
print("Test:", len(test_indices))


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading RF...")

rf_model = joblib.load(
    RF_MODEL
)

rf_imputer = joblib.load(
    RF_IMPUTER
)


print("Loading CNN V2...")

cnn_v2 = tf.keras.models.load_model(
    CNN_V2_MODEL
)

with open(
    CNN_V2_SCALER,
    "rb"
) as f:
    cnn_scaler = pickle.load(f)


print("Loading Hybrid CNN...")

hybrid_model = tf.keras.models.load_model(
    HYBRID_MODEL
)

with open(
    HYBRID_SEQ_SCALER,
    "rb"
) as f:
    hybrid_seq_scaler = pickle.load(f)

with open(
    HYBRID_PHYS_SCALER,
    "rb"
) as f:
    hybrid_phys_scaler = pickle.load(f)


# ============================================================
# BUILD RF FEATURES
# ============================================================

def build_rf_features(X):

    features = []

    for seq in X:

        row = []

        for channel in range(
            seq.shape[1]
        ):

            values = seq[:, channel]

            row.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values)
            ])

        features.append(row)

    return np.asarray(
        features,
        dtype=np.float32
    )


print("\nBuilding RF features...")

X_rf = build_rf_features(
    X_seq
)

print(
    "RF feature shape:",
    X_rf.shape
)


# ============================================================
# TEST DATA
# ============================================================

X_rf_test = X_rf[
    test_indices
]

X_seq_test = X_seq[
    test_indices
]

X_hybrid_seq_test = X_hybrid_seq[
    test_indices
]

X_physics_test = X_physics[
    test_indices
]

y_test = y[
    test_indices
]


# ============================================================
# RF PREDICTION
# ============================================================

print("\nGenerating RF predictions...")

X_rf_test_imp = rf_imputer.transform(
    X_rf_test
)

rf_pred = np.asarray(
    rf_model.predict(
        X_rf_test_imp
    ),
    dtype=np.float32
)

print(
    "RF prediction shape:",
    rf_pred.shape
)


# ============================================================
# CNN V2 PREDICTION
# ============================================================

print("Generating CNN V2 predictions...")

original_shape = X_seq_test.shape

# Scaler expects 12 sensor channels,
# not flattened 60 x 12 input.

X_cnn_scaled = cnn_scaler.transform(
    X_seq_test.reshape(
        -1,
        X_seq_test.shape[2]
    )
)

X_cnn_test = X_cnn_scaled.reshape(
    original_shape
)

cnn_v2_pred = np.asarray(
    cnn_v2.predict(
        X_cnn_test,
        verbose=0
    ),
    dtype=np.float32
)

print(
    "CNN V2 prediction shape:",
    cnn_v2_pred.shape
)


# ============================================================
# HYBRID CNN PREDICTION
# ============================================================

print("Generating Hybrid CNN predictions...")

hybrid_shape = X_hybrid_seq_test.shape

hybrid_seq_scaled = hybrid_seq_scaler.transform(
    X_hybrid_seq_test.reshape(
        -1,
        X_hybrid_seq_test.shape[2]
    )
).reshape(
    hybrid_shape
)

physics_scaled = hybrid_phys_scaler.transform(
    X_physics_test
)

hybrid_pred = np.asarray(
    hybrid_model.predict(
        [
            hybrid_seq_scaled,
            physics_scaled
        ],
        verbose=0
    ),
    dtype=np.float32
)

print(
    "Hybrid prediction shape:",
    hybrid_pred.shape
)


# ============================================================
# VALIDATE PREDICTION SHAPES
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION VALIDATION")
print("=" * 70)

print("RF V2       :", rf_pred.shape)
print("CNN V2      :", cnn_v2_pred.shape)
print("Hybrid CNN  :", hybrid_pred.shape)
print("Ground Truth:", y_test.shape)

assert rf_pred.shape == y_test.shape

assert cnn_v2_pred.shape == y_test.shape

assert hybrid_pred.shape == y_test.shape

print("\n✓ All prediction shapes match.")


# ============================================================
# ERROR CALCULATION
# ============================================================

def calculate_errors(
    prediction,
    true
):

    delta = (
        prediction -
        true
    )

    distance_error = np.sqrt(
        delta[:, 0] ** 2 +
        delta[:, 1] ** 2
    )

    return {
        "mean": np.mean(
            distance_error
        ),
        "median": np.median(
            distance_error
        ),
        "p90": np.percentile(
            distance_error,
            90
        ),
        "max": np.max(
            distance_error
        )
    }


# ============================================================
# MODEL DICTIONARY
# ============================================================

models = {

    "RF V2":
        rf_pred,

    "CNN V2":
        cnn_v2_pred,

    "Hybrid CNN V1":
        hybrid_pred
}


# ============================================================
# SINGLE STEP TEST
# ============================================================

print("\n" + "=" * 70)
print("SINGLE-STEP TEST")
print("=" * 70)

single_results = {}

for name, prediction in models.items():

    result = calculate_errors(
        prediction,
        y_test
    )

    single_results[name] = result

    print("\n" + name)

    print(
        f"Mean   : {result['mean']:.4f} m"
    )

    print(
        f"Median : {result['median']:.4f} m"
    )

    print(
        f"P90    : {result['p90']:.4f} m"
    )

    print(
        f"Maximum: {result['max']:.4f} m"
    )


# ============================================================
# STATEFUL OUTAGE TEST
# ============================================================

def stateful_outage_test(
    predictions,
    targets,
    horizon_intervals
):

    errors = []

    max_start = (
        len(predictions) -
        horizon_intervals +
        1
    )

    for start in range(
        max_start
    ):

        estimated_position = np.array(
            [0.0, 0.0],
            dtype=np.float64
        )

        true_position = np.array(
            [0.0, 0.0],
            dtype=np.float64
        )

        for step in range(
            horizon_intervals
        ):

            idx = (
                start +
                step
            )

            estimated_position += (
                predictions[idx]
            )

            true_position += (
                targets[idx]
            )

        error = np.linalg.norm(
            estimated_position -
            true_position
        )

        errors.append(
            error
        )

    errors = np.asarray(
        errors
    )

    return {

        "trials":
            len(errors),

        "mean":
            np.mean(errors),

        "median":
            np.median(errors),

        "p90":
            np.percentile(
                errors,
                90
            ),

        "max":
            np.max(errors)
    }


# ============================================================
# OUTAGE HORIZONS
# ============================================================

horizons = {

    "10 sec": 1,

    "20 sec": 2,

    "30 sec": 3,

    "60 sec": 6
}


# ============================================================
# RUN UNIFIED BENCHMARK
# ============================================================

print("\n" + "=" * 70)
print("STATEFUL GNSS OUTAGE BENCHMARK")
print("=" * 70)

all_results = []


for duration, intervals in horizons.items():

    print(
        f"\n{'-' * 60}"
    )

    print(
        f"OUTAGE: {duration}"
    )

    print(
        f"Intervals: {intervals}"
    )

    print(
        f"{'-' * 60}"
    )

    for name, prediction in models.items():

        result = stateful_outage_test(
            prediction,
            y_test,
            intervals
        )

        print(
            f"{name:18s} | "
            f"Mean {result['mean']:8.3f} m | "
            f"Median {result['median']:8.3f} m | "
            f"P90 {result['p90']:8.3f} m | "
            f"Max {result['max']:8.3f} m"
        )

        all_results.append({

            "model":
                name,

            "outage":
                duration,

            "intervals":
                intervals,

            "trials":
                result["trials"],

            "mean":
                result["mean"],

            "median":
                result["median"],

            "p90":
                result["p90"],

            "max":
                result["max"]
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

output_file = os.path.join(
    DATA_DIR,
    "unified_outage_benchmark_v1.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL RANKING
# ============================================================

print("\n" + "=" * 70)
print("FINAL STATEFUL RANKING")
print("=" * 70)


for duration in horizons:

    subset = results_df[
        results_df["outage"] ==
        duration
    ].sort_values(
        "mean"
    )

    print(
        f"\n{duration}"
    )

    for rank, (_, row) in enumerate(
        subset.iterrows(),
        start=1
    ):

        print(
            f"{rank}. "
            f"{row['model']:18s} "
            f"{row['mean']:.3f} m"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 35 COMPLETE")
print("=" * 70)

print("\nSaved:")

print(
    output_file
)