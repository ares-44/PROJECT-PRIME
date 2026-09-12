import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PROJECT PRIME
# Navigation Engine V1
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# FILES
# ============================================================

DATASET = os.path.join(
    DATA_DIR,
    "cleaned_S-S1.csv"
)

SEQUENCE_DATA = os.path.join(
    DATA_DIR,
    "prime_sequence_dataset_v2.npz"
)

CNN_MODEL = os.path.join(
    MODEL_DIR,
    "prime_temporal_cnn_v2.keras"
)

CNN_SCALER = os.path.join(
    MODEL_DIR,
    "prime_sequence_scaler_v2.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Dataset sensor interval
DT = 0.1

# Number of sensor samples used by CNN V2
SEQUENCE_LENGTH = 60

# Approximately one GNSS fix every ~9 seconds
GNSS_FIX_INTERVAL = 9.0

# Simulated outage durations
OUTAGE_DURATIONS = [
    10,
    20,
    30,
    60
]


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("=" * 70)
print("PROJECT PRIME — NAVIGATION ENGINE V1")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET
)

print(
    "Dataset rows:",
    len(df)
)


# ============================================================
# LOAD SEQUENCE DATA
# ============================================================

print("\nLoading sequence dataset...")

seq_data = np.load(
    SEQUENCE_DATA
)

X = seq_data["X"].astype(
    np.float32
)

y = seq_data["y"].astype(
    np.float32
)

print(
    "Sequence shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)


# ============================================================
# LOAD CNN V2
# ============================================================

print("\nLoading CNN V2...")

model = tf.keras.models.load_model(
    CNN_MODEL
)

with open(
    CNN_SCALER,
    "rb"
) as f:
    scaler = pickle.load(f)

print("CNN V2 loaded successfully.")


# ============================================================
# PREPARE CNN INPUT
# ============================================================

def prepare_sequence(sequence):

    shape = sequence.shape

    # scaler expects 12 sensor channels
    scaled = scaler.transform(
        sequence.reshape(
            -1,
            sequence.shape[1]
        )
    )

    scaled = scaled.reshape(
        shape
    )

    return scaled


# ============================================================
# PREDICT DISPLACEMENT
# ============================================================

def predict_displacement(sequence):

    X_input = prepare_sequence(
        sequence
    )

    prediction = model.predict(
        X_input[np.newaxis, ...],
        verbose=0
    )

    return prediction[0]


# ============================================================
# LOCAL POSITION → LAT/LON
# ============================================================

def displacement_to_latlon(
    north_m,
    east_m,
    lat0,
    lon0
):

    meters_per_degree_lat = 111320.0

    meters_per_degree_lon = (
        111320.0 *
        np.cos(
            np.radians(lat0)
        )
    )

    lat = (
        lat0 +
        north_m /
        meters_per_degree_lat
    )

    lon = (
        lon0 +
        east_m /
        meters_per_degree_lon
    )

    return lat, lon


# ============================================================
# BUILD TEST SEQUENCE
# ============================================================

N = len(X)

train_end = int(
    N * 0.70
)

val_end = int(
    N * 0.85
)

test_start = val_end


# ============================================================
# NAVIGATION SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("PRIME NAVIGATION SIMULATION")
print("=" * 70)

print(
    "\nMode sequence:"
)

print(
    "GNSS_NORMAL → GNSS_OUTAGE → GNSS_RECOVERY"
)


# ============================================================
# USE FIRST TEST SAMPLE AS START
# ============================================================

start_idx = test_start

# Starting position is represented as
# a local coordinate system.

estimated_north = 0.0
estimated_east = 0.0

true_north = 0.0
true_east = 0.0


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RUN OUTAGE SIMULATIONS
# ============================================================

for outage_seconds in OUTAGE_DURATIONS:

    print("\n" + "-" * 70)

    print(
        f"GNSS OUTAGE: {outage_seconds} seconds"
    )

    print("-" * 70)

    # Approximate number of GNSS intervals
    intervals = max(
        1,
        round(
            outage_seconds /
            GNSS_FIX_INTERVAL
        )
    )

    print(
        "Prediction intervals:",
        intervals
    )

    # Reset state
    estimated_north = 0.0
    estimated_east = 0.0

    true_north = 0.0
    true_east = 0.0

    # Store trajectory
    trajectory = []

    # --------------------------------------------------------
    # GNSS NORMAL
    # --------------------------------------------------------

    mode = "GNSS_NORMAL"

    print(
        "\nMode:",
        mode
    )

    print(
        "Initial position locked."
    )

    # --------------------------------------------------------
    # GNSS OUTAGE
    # --------------------------------------------------------

    mode = "GNSS_OUTAGE"

    print(
        "\nMode:",
        mode
    )

    for step in range(
        intervals
    ):

        idx = start_idx + step

        if idx >= N:
            break

        # --------------------------------------------
        # Sensor history
        # --------------------------------------------

        sequence = X[idx]

        # --------------------------------------------
        # CNN prediction
        # --------------------------------------------

        predicted_delta = predict_displacement(
            sequence
        )

        predicted_north = float(
            predicted_delta[0]
        )

        predicted_east = float(
            predicted_delta[1]
        )

        # --------------------------------------------
        # Ground truth displacement
        # --------------------------------------------

        actual_delta = y[idx]

        actual_north = float(
            actual_delta[0]
        )

        actual_east = float(
            actual_delta[1]
        )

        # --------------------------------------------
        # Update PRIME position
        # --------------------------------------------

        estimated_north += (
            predicted_north
        )

        estimated_east += (
            predicted_east
        )

        # --------------------------------------------
        # Update true position
        # --------------------------------------------

        true_north += (
            actual_north
        )

        true_east += (
            actual_east
        )

        # --------------------------------------------
        # Position error
        # --------------------------------------------

        error = np.sqrt(
            (
                estimated_north -
                true_north
            ) ** 2
            +
            (
                estimated_east -
                true_east
            ) ** 2
        )

        trajectory.append({

            "step":
                step + 1,

            "elapsed_seconds":
                (step + 1) *
                GNSS_FIX_INTERVAL,

            "predicted_north":
                predicted_north,

            "predicted_east":
                predicted_east,

            "estimated_north":
                estimated_north,

            "estimated_east":
                estimated_east,

            "true_north":
                true_north,

            "true_east":
                true_east,

            "position_error":
                error
        })

        print(
            f"Step {step + 1:2d} | "
            f"Pred ΔN={predicted_north:8.2f} m | "
            f"Pred ΔE={predicted_east:8.2f} m | "
            f"Error={error:8.2f} m"
        )

    # --------------------------------------------------------
    # GNSS RECOVERY
    # --------------------------------------------------------

    mode = "GNSS_RECOVERY"

    print(
        "\nMode:",
        mode
    )

    final_error = np.sqrt(
        (
            estimated_north -
            true_north
        ) ** 2
        +
        (
            estimated_east -
            true_east
        ) ** 2
    )

    print(
        f"PRIME position error before "
        f"GNSS recovery: {final_error:.2f} m"
    )

    # GNSS recovery resets the navigation state
    estimated_north = true_north
    estimated_east = true_east

    recovery_error = np.sqrt(
        (
            estimated_north -
            true_north
        ) ** 2
        +
        (
            estimated_east -
            true_east
        ) ** 2
    )

    print(
        f"Position error after "
        f"GNSS recovery: {recovery_error:.2f} m"
    )

    results.append({

        "outage_seconds":
            outage_seconds,

        "intervals":
            intervals,

        "final_error_before_recovery":
            final_error,

        "error_after_recovery":
            recovery_error
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_file = os.path.join(
    DATA_DIR,
    "prime_navigation_engine_v1_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PRIME NAVIGATION ENGINE — FINAL SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(
    output_file
)

print("\n" + "=" * 70)
print("STEP 36 COMPLETE")
print("=" * 70)