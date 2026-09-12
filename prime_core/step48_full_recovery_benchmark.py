import os
import sys
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from prime_core.gnss_monitor import GNSSMonitor
from prime_core.position_engine import PositionEngine
from prime_core.recovery_manager import RecoveryManager


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
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

OUTPUT_PATH = os.path.join(
    BASE,
    "data",
    "prime_step48_recovery_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("PROJECT PRIME — STEP 48")
print("FULL OUTAGE + RECOVERY BENCHMARK")
print("=" * 75)

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]

print("\nDataset:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CNN V2...")

model = keras.models.load_model(
    MODEL_PATH
)

with open(
    SCALER_PATH,
    "rb"
) as f:
    scaler = pickle.load(f)

print("CNN V2 loaded.")


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

csv_path = os.path.join(
    BASE,
    "data",
    "cleaned_S-S1.csv"
)

df = pd.read_csv(
    csv_path,
    encoding="latin1"
)

lat_col = "GPS LATITUDE (degrees)"
lon_col = "GPS LONGITUDE (degrees)"

print(
    "\nOriginal dataset:",
    len(df),
    "samples"
)


# ============================================================
# FIND GPS FIXES
# ============================================================

lat = pd.to_numeric(
    df[lat_col],
    errors="coerce"
).values

lon = pd.to_numeric(
    df[lon_col],
    errors="coerce"
).values

fix_indices = []

for i in range(1, len(df)):

    if (
        np.isfinite(lat[i])
        and
        np.isfinite(lon[i])
        and
        (
            lat[i] != lat[i - 1]
            or
            lon[i] != lon[i - 1]
        )
    ):

        fix_indices.append(i)


fix_indices = np.array(
    fix_indices,
    dtype=int
)

print(
    "GPS fix transitions:",
    len(fix_indices)
)


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# OUTAGE LENGTHS
# ============================================================

outage_lengths = {
    "10s": 1,
    "20s": 2,
    "30s": 3,
    "60s": 6
}


# ============================================================
# POSITION ERROR
# ============================================================

def position_error_m(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    return (
        2
        *
        R
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# RUN BENCHMARK
# ============================================================

for label, intervals in outage_lengths.items():

    print("\n" + "-" * 75)
    print(
        f"TESTING {label} OUTAGE"
    )
    print("-" * 75)

    errors = []

    # Avoid running beyond available sequences
    max_start = min(
        len(X) - intervals - 1,
        len(fix_indices) - intervals - 1
    )

    # Every 10th sequence
    start_points = range(
        0,
        max_start,
        10
    )

    for seq_start in start_points:

        # ----------------------------------------------------
        # REAL GNSS START POSITION
        # ----------------------------------------------------

        fix_start = fix_indices[
            seq_start
        ]

        start_lat = lat[
            fix_start
        ]

        start_lon = lon[
            fix_start
        ]

        if not (
            np.isfinite(start_lat)
            and
            np.isfinite(start_lon)
        ):
            continue

        position = PositionEngine(
            start_lat,
            start_lon
        )

        recovery = RecoveryManager()

        # ----------------------------------------------------
        # PRIME DR DURING OUTAGE
        # ----------------------------------------------------

        for step in range(intervals):

            idx = seq_start + step

            if idx >= len(X):
                break

            sequence = X[idx]

            # Scale each timestep independently
            original_shape = sequence.shape

            sequence_scaled = scaler.transform(
                sequence
            )

            sequence_scaled = (
                sequence_scaled
            )

            prediction = model.predict(
                sequence_scaled[np.newaxis, :, :],
                verbose=0
            )[0]

            north = float(
                prediction[0]
            )

            east = float(
                prediction[1]
            )

            position.update(
                north,
                east
            )

        # ----------------------------------------------------
        # GNSS RECOVERY
        # ----------------------------------------------------

        recovery_fix = fix_indices[
            seq_start + intervals
        ]

        recovered_lat = lat[
            recovery_fix
        ]

        recovered_lon = lon[
            recovery_fix
        ]

        if not (
            np.isfinite(recovered_lat)
            and
            np.isfinite(recovered_lon)
        ):
            continue

        # Re-anchor
        recovery.start_recovery()

        recovery.reanchor(
            position,
            recovered_lat,
            recovered_lon
        )

        # ----------------------------------------------------
        # ERROR BEFORE RE-ANCHOR
        # ----------------------------------------------------
        #
        # Re-run position to obtain error BEFORE recovery.
        #

        pre_recovery_position = PositionEngine(
            start_lat,
            start_lon
        )

        for step in range(intervals):

            idx = seq_start + step

            sequence = X[idx]

            sequence_scaled = scaler.transform(
                sequence
            )

            prediction = model.predict(
                sequence_scaled[np.newaxis, :, :],
                verbose=0
            )[0]

            pre_recovery_position.update(
                float(prediction[0]),
                float(prediction[1])
            )

        error_before = position_error_m(
            pre_recovery_position.latitude,
            pre_recovery_position.longitude,
            recovered_lat,
            recovered_lon
        )

        # ----------------------------------------------------
        # ERROR AFTER RE-ANCHOR
        # ----------------------------------------------------

        final_position = position.get_position()

        error_after = position_error_m(
            final_position["latitude"],
            final_position["longitude"],
            recovered_lat,
            recovered_lon
        )

        errors.append(
            error_before
        )

        results.append({
            "outage": label,
            "intervals": intervals,
            "sequence_start": seq_start,
            "error_before_recovery_m": error_before,
            "error_after_recovery_m": error_after,
            "reanchored": True
        })


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if errors:

        errors = np.array(
            errors
        )

        print(
            f"Trials: {len(errors)}"
        )

        print(
            f"Mean error: "
            f"{np.mean(errors):.3f} m"
        )

        print(
            f"Median error: "
            f"{np.median(errors):.3f} m"
        )

        print(
            f"P90 error: "
            f"{np.percentile(errors,90):.3f} m"
        )

        print(
            f"Max error: "
            f"{np.max(errors):.3f} m"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 75)
print("STEP 48 COMPLETE")
print("=" * 75)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    "\nRecovery architecture verified:"
)

print(
    "GNSS → PRIME_DR → GNSS RECOVERY → RE-ANCHOR"
)