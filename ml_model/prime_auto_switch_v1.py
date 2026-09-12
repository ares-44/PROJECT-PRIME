import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PROJECT PRIME
# AUTO SWITCH ENGINE V1
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# FILES
# ============================================================

HEALTH_FILE = os.path.join(
    DATA_DIR,
    "gnss_health_monitor_v1_results.csv"
)

SEQUENCE_FILE = os.path.join(
    DATA_DIR,
    "prime_sequence_dataset_v2.npz"
)

CNN_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "prime_temporal_cnn_v2.keras"
)

CNN_SCALER_FILE = os.path.join(
    MODEL_DIR,
    "prime_sequence_scaler_v2.pkl"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "prime_auto_switch_v1_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60


# ============================================================
# LOAD GNSS HEALTH
# ============================================================

print("=" * 70)
print("PROJECT PRIME — AUTO SWITCH ENGINE V1")
print("=" * 70)

print("\nLoading GNSS health data...")

health_df = pd.read_csv(
    HEALTH_FILE
)

print(
    "Health samples:",
    len(health_df)
)


# ============================================================
# LOAD SENSOR SEQUENCES
# ============================================================

print("\nLoading sensor sequences...")

seq_data = np.load(
    SEQUENCE_FILE
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
    CNN_MODEL_FILE
)

with open(
    CNN_SCALER_FILE,
    "rb"
) as f:

    scaler = pickle.load(f)

print(
    "CNN V2 loaded."
)


# ============================================================
# PREPARE CNN INPUT
# ============================================================

def prepare_sequence(sequence):

    original_shape = sequence.shape

    scaled = scaler.transform(
        sequence.reshape(
            -1,
            sequence.shape[1]
        )
    )

    return scaled.reshape(
        original_shape
    )


# ============================================================
# CNN DISPLACEMENT PREDICTION
# ============================================================

def predict_displacement(sequence):

    prepared = prepare_sequence(
        sequence
    )

    prediction = model.predict(
        prepared[np.newaxis, ...],
        verbose=0
    )

    return prediction[0]


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

# Sequence dataset contains one sample approximately
# every GNSS fix interval.

health_states = health_df[
    "gnss_state"
].values


# ============================================================
# MAP SEQUENCE INDEX → HEALTH SAMPLE
# ============================================================

# The sequence dataset contains 513 valid GNSS intervals,
# while the health monitor contains all sensor samples.
#
# We therefore use the approximately 9-second GNSS interval
# to locate the corresponding health state.

time_seconds = health_df[
    "time_seconds"
].values


# ============================================================
# CREATE GNSS STATE AT EACH SEQUENCE
# ============================================================

sequence_times = []

for i in range(
    len(X)
):

    # Approximate fix interval
    sequence_times.append(
        (i + 1) * 9.0
    )

sequence_times = np.asarray(
    sequence_times
)


def get_health_state(
    time_value
):

    idx = np.searchsorted(
        time_seconds,
        time_value
    )

    if idx >= len(
        health_states
    ):

        idx = (
            len(health_states)
            - 1
        )

    return health_states[idx]


# ============================================================
# AUTO SWITCH SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATIC GNSS / PRIME SWITCHING")
print("=" * 70)


estimated_north = 0.0
estimated_east = 0.0

true_north = 0.0
true_east = 0.0

previous_state = None

results = []

prime_active = False


# ============================================================
# PROCESS SEQUENCES
# ============================================================

for i in range(
    len(X)
):

    state = get_health_state(
        sequence_times[i]
    )

    # --------------------------------------------------------
    # STATE TRANSITION
    # --------------------------------------------------------

    if state != previous_state:

        print(
            f"\nTime "
            f"{sequence_times[i]:8.2f}s"
        )

        print(
            f"State: "
            f"{previous_state} "
            f"→ {state}"
        )

        previous_state = state


    # --------------------------------------------------------
    # GNSS GOOD
    # --------------------------------------------------------

    if state == "GNSS_GOOD":

        prime_active = False

        # GNSS is trusted.
        # Reset estimated state to true GNSS state
        # in this offline simulation.

        estimated_north = true_north
        estimated_east = true_east

        mode = "GNSS"


    # --------------------------------------------------------
    # GNSS DEGRADED
    # --------------------------------------------------------

    elif state == "GNSS_DEGRADED":

        # Continue monitoring.
        # PRIME does not take over yet.

        mode = "GNSS_DEGRADED"


    # --------------------------------------------------------
    # GNSS OUTAGE
    # --------------------------------------------------------

    else:

        if not prime_active:

            print(
                ">>> PRIME CNN V2 ACTIVATED"
            )

            prime_active = True

        mode = "PRIME_DR"

        # CNN prediction
        prediction = predict_displacement(
            X[i]
        )

        estimated_north += float(
            prediction[0]
        )

        estimated_east += float(
            prediction[1]
        )


    # --------------------------------------------------------
    # TRUE STATE
    # --------------------------------------------------------

    true_north += float(
        y[i, 0]
    )

    true_east += float(
        y[i, 1]
    )


    # --------------------------------------------------------
    # POSITION ERROR
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append({

        "sequence_index":
            i,

        "time_seconds":
            sequence_times[i],

        "gnss_state":
            state,

        "navigation_mode":
            mode,

        "prime_active":
            prime_active,

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


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUTO SWITCH SUMMARY")
print("=" * 70)

print(
    "\nNavigation modes:"
)

print(
    results_df[
        "navigation_mode"
    ].value_counts()
)


print(
    "\nPRIME activation events:"
)

activation_count = 0

previous = False

for active in results_df[
    "prime_active"
]:

    if active and not previous:

        activation_count += 1

    previous = active


print(
    activation_count
)


print(
    "\nMaximum PRIME position error:"
)

print(
    f"{results_df['position_error'].max():.3f} m"
)


print(
    "\nMean position error:"
)

print(
    f"{results_df['position_error'].mean():.3f} m"
)


# ============================================================
# SAVE
# ============================================================

print("\nSaved:")
print(
    OUTPUT_FILE
)

print("\n" + "=" * 70)
print("STEP 38 COMPLETE")
print("=" * 70)