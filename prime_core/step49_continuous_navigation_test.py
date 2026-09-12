import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

CSV_PATH = os.path.join(
    BASE,
    "data",
    "cleaned_S-S1.csv"
)

SEQ_PATH = os.path.join(
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

RESULT_PATH = os.path.join(
    BASE,
    "data",
    "prime_step49_continuous_results.csv"
)

PLOT_PATH = os.path.join(
    BASE,
    "data",
    "prime_step49_continuous_trajectory.png"
)

ERROR_PLOT_PATH = os.path.join(
    BASE,
    "data",
    "prime_step49_continuous_error.png"
)


print("=" * 75)
print("PROJECT PRIME — STEP 49")
print("CONTINUOUS NAVIGATION + REPEATED GNSS RECOVERY")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    encoding="latin1"
)

data = np.load(
    SEQ_PATH
)

X = data["X"]

print("\nOriginal samples:", len(df))
print("Sequence samples:", len(X))


# ============================================================
# MODEL
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
# GPS COLUMNS
# ============================================================

lat_col = "GPS LATITUDE (degrees)"
lon_col = "GPS LONGITUDE (degrees)"

lat = pd.to_numeric(
    df[lat_col],
    errors="coerce"
).values

lon = pd.to_numeric(
    df[lon_col],
    errors="coerce"
).values


# ============================================================
# FIND GPS FIX TRANSITIONS
# ============================================================

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
# GEO DISTANCE
# ============================================================

def distance_m(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dp = p2 - p1
    dl = np.radians(lon2 - lon1)

    a = (
        np.sin(dp / 2) ** 2
        +
        np.cos(p1)
        *
        np.cos(p2)
        *
        np.sin(dl / 2) ** 2
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
# INITIALIZE
# ============================================================

start_index = fix_indices[0]

prime_position = PositionEngine(
    lat[start_index],
    lon[start_index]
)

monitor = GNSSMonitor()
recovery = RecoveryManager()


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

# Repeated artificial outages.
#
# These are deliberately deterministic so the experiment
# can be reproduced.

outage_pattern = [
    1,  # ~10 sec
    2,  # ~20 sec
    1,  # ~10 sec
    3,  # ~30 sec
    2,  # ~20 sec
    1,  # ~10 sec
    6,  # ~60 sec
    2,  # ~20 sec
    3,  # ~30 sec
    1   # ~10 sec
]


# ============================================================
# STORAGE
# ============================================================

records = []

outage_events = []

recovery_events = []

total_prime_steps = 0

total_gnss_steps = 0

max_error = 0.0


# ============================================================
# CONTINUOUS JOURNEY
# ============================================================

seq_index = 0

pattern_index = 0

current_outage_remaining = 0

in_outage = False


while (
    seq_index < len(X)
    and
    seq_index < len(fix_indices) - 1
):

    current_fix = fix_indices[
        seq_index
    ]

    next_fix = fix_indices[
        seq_index + 1
    ]

    true_lat = lat[next_fix]
    true_lon = lon[next_fix]

    if not (
        np.isfinite(true_lat)
        and
        np.isfinite(true_lon)
    ):
        seq_index += 1
        continue

    # --------------------------------------------------------
    # START NEXT ARTIFICIAL OUTAGE
    # --------------------------------------------------------

    if not in_outage:

        if pattern_index < len(outage_pattern):

            current_outage_remaining = (
                outage_pattern[pattern_index]
            )

            pattern_index += 1

            in_outage = True

            outage_events.append({
                "sequence_start": seq_index,
                "duration_intervals":
                    current_outage_remaining
            })

        else:

            # No more artificial outages.
            # Continue using GNSS.

            in_outage = False


    # --------------------------------------------------------
    # PRIME OUTAGE
    # --------------------------------------------------------

    if in_outage:

        sequence = X[seq_index]

        sequence_scaled = scaler.transform(
            sequence
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

        prime_position.update(
            north,
            east
        )

        total_prime_steps += 1

        current_position = (
            prime_position.get_position()
        )

        error = distance_m(
            current_position["latitude"],
            current_position["longitude"],
            true_lat,
            true_lon
        )

        max_error = max(
            max_error,
            error
        )

        records.append({
            "sequence": seq_index,
            "mode": "PRIME_DR",
            "latitude":
                current_position["latitude"],
            "longitude":
                current_position["longitude"],
            "true_latitude":
                true_lat,
            "true_longitude":
                true_lon,
            "error_m": error
        })

        current_outage_remaining -= 1

        if current_outage_remaining <= 0:

            in_outage = False

            # ------------------------------------------------
            # GNSS RECOVERY
            # ------------------------------------------------

            recovery.start_recovery()

            recovered = recovery.reanchor(
                prime_position,
                true_lat,
                true_lon
            )

            recovery_events.append({
                "sequence":
                    seq_index,
                "reanchored":
                    recovered,
                "error_before_recovery_m":
                    error,
                "error_after_recovery_m":
                    0.0
            })

    # --------------------------------------------------------
    # GNSS NORMAL
    # --------------------------------------------------------

    else:

        # Re-anchor continuously while GNSS is available.
        prime_position.reset(
            true_lat,
            true_lon
        )

        total_gnss_steps += 1

        records.append({
            "sequence": seq_index,
            "mode": "GNSS",
            "latitude":
                true_lat,
            "longitude":
                true_lon,
            "true_latitude":
                true_lat,
            "true_longitude":
                true_lon,
            "error_m": 0.0
        })

    seq_index += 1


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    records
)

results.to_csv(
    RESULT_PATH,
    index=False
)


prime_results = results[
    results["mode"] == "PRIME_DR"
]

errors = prime_results[
    "error_m"
].values


print("\n" + "=" * 75)
print("STEP 49 RESULTS")
print("=" * 75)

print(
    "\nGNSS steps:",
    total_gnss_steps
)

print(
    "PRIME DR steps:",
    total_prime_steps
)

print(
    "Outage events:",
    len(outage_events)
)

print(
    "Recovery events:",
    len(recovery_events)
)

if len(errors) > 0:

    print(
        "\nPRIME outage error:"
    )

    print(
        f"Mean:   {np.mean(errors):.3f} m"
    )

    print(
        f"Median: {np.median(errors):.3f} m"
    )

    print(
        f"P90:    {np.percentile(errors,90):.3f} m"
    )

    print(
        f"Max:    {np.max(errors):.3f} m"
    )


# ============================================================
# RECOVERY REPORT
# ============================================================

print(
    "\nRECOVERY EVENTS"
)

for i, event in enumerate(
    recovery_events,
    start=1
):

    print(
        f"Recovery #{i}: "
        f"before="
        f"{event['error_before_recovery_m']:.2f} m "
        f"after="
        f"{event['error_after_recovery_m']:.2f} m"
    )


# ============================================================
# TRAJECTORY PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    results["true_longitude"],
    results["true_latitude"],
    label="Reference trajectory"
)

prime_mask = (
    results["mode"]
    == "PRIME_DR"
)

plt.scatter(
    results.loc[
        prime_mask,
        "longitude"
    ],
    results.loc[
        prime_mask,
        "latitude"
    ],
    s=8,
    label="PRIME DR"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "PROJECT PRIME — Continuous Navigation"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=200
)

plt.close()


# ============================================================
# ERROR PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    results["sequence"],
    results["error_m"]
)

plt.xlabel(
    "Sequence"
)

plt.ylabel(
    "Position Error (m)"
)

plt.title(
    "PROJECT PRIME — Continuous Position Error"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    ERROR_PLOT_PATH,
    dpi=200
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\nSaved:")

print(
    RESULT_PATH
)

print(
    PLOT_PATH
)

print(
    ERROR_PLOT_PATH
)

print(
    "\n" + "=" * 75
)

print(
    "STEP 49 COMPLETE"
)

print(
    "GNSS → PRIME_DR → RECOVERY → RE-ANCHOR"
)

print("=" * 75)