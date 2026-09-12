import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from prime_core.inference_engine import PRIMEInferenceEngine
from prime_core.position_engine import PositionEngine


# ============================================================
# PROJECT PRIME — STEP 46
# REALISTIC GNSS OUTAGE → RECOVERY BENCHMARK
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CSV_FILE = os.path.join(
    DATA_DIR,
    "cleaned_S-S1.csv"
)

SEQUENCE_FILE = os.path.join(
    DATA_DIR,
    "prime_sequence_dataset_v2.npz"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "prime_temporal_cnn_v2.keras"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "prime_sequence_scaler_v2.pkl"
)

OUTPUT_CSV = os.path.join(
    DATA_DIR,
    "prime_outage_recovery_benchmark_v1.csv"
)

OUTPUT_TRAJECTORY = os.path.join(
    DATA_DIR,
    "prime_outage_recovery_trajectory_v1.png"
)

OUTPUT_ERROR = os.path.join(
    DATA_DIR,
    "prime_outage_recovery_error_v1.png"
)


print("=" * 70)
print("PROJECT PRIME — STEP 46")
print("REALISTIC OUTAGE → RECOVERY BENCHMARK")
print("=" * 70)


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

print("\nLoading original dataset...")

df = pd.read_csv(
    CSV_FILE,
    encoding="latin1"
)

lat_col = "GPS LATITUDE (degrees)"
lon_col = "GPS LONGITUDE (degrees)"

lat = pd.to_numeric(
    df[lat_col],
    errors="coerce"
)

lon = pd.to_numeric(
    df[lon_col],
    errors="coerce"
)


# ============================================================
# 2. FIND GPS FIXES
# ============================================================

fix_change = (
    lat.ne(lat.shift())
    |
    lon.ne(lon.shift())
)

fix_indices = np.where(
    fix_change
)[0]

print(
    "GPS fix transitions:",
    len(fix_indices)
)


# ============================================================
# 3. LOAD SEQUENCE DATA
# ============================================================

print("\nLoading sequence dataset...")

seq_data = np.load(
    SEQUENCE_FILE
)

X = seq_data["X"]

print(
    "Sequence shape:",
    X.shape
)


# ============================================================
# 4. LOAD CNN V2
# ============================================================

print("\nLoading CNN V2...")

engine = PRIMEInferenceEngine(
    MODEL_FILE,
    SCALER_FILE
)

print("CNN V2 loaded.")


# ============================================================
# 5. TEST CONFIGURATION
# ============================================================

# Dataset GPS interval is approximately 9 seconds.
#
# Therefore:
#
# 1 interval ≈ 10 sec
# 2 intervals ≈ 20 sec
# 3 intervals ≈ 30 sec
# 6 intervals ≈ 60 sec

OUTAGE_CONFIGS = {
    "10 sec": 1,
    "20 sec": 2,
    "30 sec": 3,
    "60 sec": 6
}


# Don't use every possible starting point.
# We space outage events so they don't overlap heavily.

START_STEP = 10


all_results = []


# ============================================================
# 6. RUN OUTAGE TESTS
# ============================================================

for outage_name, outage_intervals in OUTAGE_CONFIGS.items():

    print("\n" + "=" * 70)

    print(
        f"TESTING {outage_name} "
        f"({outage_intervals} GPS intervals)"
    )

    print("=" * 70)

    errors = []

    for start in range(
        0,
        len(X) - outage_intervals,
        START_STEP
    ):

        # ----------------------------------------------------
        # Initial real GPS position
        # ----------------------------------------------------

        start_idx = fix_indices[start]

        start_lat = float(
            lat.iloc[start_idx]
        )

        start_lon = float(
            lon.iloc[start_idx]
        )

        position_engine = PositionEngine(
            start_lat,
            start_lon
        )

        # ----------------------------------------------------
        # PRIME outage simulation
        # ----------------------------------------------------

        for step in range(
            outage_intervals
        ):

            sequence_idx = (
                start + step
            )

            prediction = engine.predict(
                X[sequence_idx]
            )

            north = float(
                prediction[0]
            )

            east = float(
                prediction[1]
            )

            position_engine.update(
                north,
                east
            )

        # ----------------------------------------------------
        # Position when GNSS returns
        # ----------------------------------------------------

        recovery_idx = (
            start + outage_intervals
        )

        if recovery_idx >= len(fix_indices):
            continue

        true_idx = fix_indices[
            recovery_idx
        ]

        true_lat = float(
            lat.iloc[true_idx]
        )

        true_lon = float(
            lon.iloc[true_idx]
        )

        prime_position = (
            position_engine.get_position()
        )

        prime_lat = prime_position[
            "latitude"
        ]

        prime_lon = prime_position[
            "longitude"
        ]

        # ----------------------------------------------------
        # Calculate error
        # ----------------------------------------------------

        mean_lat = np.radians(
            (true_lat + prime_lat) / 2
        )

        north_error = (
            prime_lat - true_lat
        ) * 111320.0

        east_error = (
            prime_lon - true_lon
        ) * 111320.0 * np.cos(
            mean_lat
        )

        error = np.sqrt(
            north_error ** 2
            +
            east_error ** 2
        )

        errors.append(
            error
        )

        all_results.append({

            "outage":
                outage_name,

            "outage_intervals":
                outage_intervals,

            "start_sequence":
                start,

            "recovery_sequence":
                recovery_idx,

            "start_latitude":
                start_lat,

            "start_longitude":
                start_lon,

            "actual_latitude":
                true_lat,

            "actual_longitude":
                true_lon,

            "prime_latitude":
                prime_lat,

            "prime_longitude":
                prime_lon,

            "position_error_m":
                error
        })


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    errors = np.array(errors)

    print(
        f"Trials  : {len(errors)}"
    )

    print(
        f"Mean    : {np.mean(errors):.3f} m"
    )

    print(
        f"Median  : {np.median(errors):.3f} m"
    )

    print(
        f"P90     : {np.percentile(errors, 90):.3f} m"
    )

    print(
        f"Maximum : {np.max(errors):.3f} m"
    )


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 46 SUMMARY")
print("=" * 70)

for outage_name in OUTAGE_CONFIGS:

    subset = results_df[
        results_df["outage"] ==
        outage_name
    ]

    errors = subset[
        "position_error_m"
    ].values

    print(
        f"\n{outage_name}"
    )

    print(
        f"Mean    : {np.mean(errors):.2f} m"
    )

    print(
        f"Median  : {np.median(errors):.2f} m"
    )

    print(
        f"P90     : {np.percentile(errors, 90):.2f} m"
    )

    print(
        f"Maximum : {np.max(errors):.2f} m"
    )


# ============================================================
# 9. TRAJECTORY VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 8)
)

# Show a subset of actual and PRIME positions.

sample = results_df[
    results_df["outage"] == "60 sec"
]

plt.plot(
    sample["actual_longitude"],
    sample["actual_latitude"],
    marker="o",
    label="Actual GNSS"
)

plt.plot(
    sample["prime_longitude"],
    sample["prime_latitude"],
    marker="x",
    label="PRIME at recovery"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "PROJECT PRIME — 60s Outage Recovery"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_TRAJECTORY,
    dpi=200
)

plt.close()


# ============================================================
# 10. ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 7)
)

labels = []
means = []

for outage_name in OUTAGE_CONFIGS:

    subset = results_df[
        results_df["outage"] ==
        outage_name
    ]

    labels.append(
        outage_name
    )

    means.append(
        subset["position_error_m"].mean()
    )

plt.bar(
    labels,
    means
)

plt.xlabel(
    "GNSS outage duration"
)

plt.ylabel(
    "Mean recovery position error (m)"
)

plt.title(
    "PROJECT PRIME — Outage Recovery Error"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_ERROR,
    dpi=200
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\nGenerated files:")

print(
    OUTPUT_CSV
)

print(
    OUTPUT_TRAJECTORY
)

print(
    OUTPUT_ERROR
)

print("\n" + "=" * 70)
print("STEP 46 COMPLETE")
print("=" * 70)