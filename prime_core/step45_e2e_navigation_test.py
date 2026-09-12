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

from prime_core.gnss_monitor import GNSSMonitor
from prime_core.inference_engine import PRIMEInferenceEngine
from prime_core.position_engine import PositionEngine
from prime_core.prime_navigation import PRIMENavigation


# ============================================================
# PROJECT PRIME — STEP 45
# REAL END-TO-END NAVIGATION TEST
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
    "prime_step45_e2e_results.csv"
)

OUTPUT_PLOT = os.path.join(
    DATA_DIR,
    "prime_step45_e2e_trajectory.png"
)


print("=" * 70)
print("PROJECT PRIME — STEP 45")
print("REAL END-TO-END NAVIGATION TEST")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading original dataset...")

df = pd.read_csv(
    CSV_FILE,
    encoding="latin1"
)

print(
    "Dataset rows:",
    len(df)
)


# ============================================================
# 2. LOAD SEQUENCE DATASET
# ============================================================

print("\nLoading sequence dataset...")

sequence_data = np.load(
    SEQUENCE_FILE
)

X = sequence_data["X"]
y = sequence_data["y"]

print(
    "Sequence shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)


# ============================================================
# 3. LOAD CNN V2
# ============================================================

print("\nLoading CNN V2...")

inference_engine = PRIMEInferenceEngine(
    MODEL_FILE,
    SCALER_FILE
)

print("CNN V2 loaded.")


# ============================================================
# 4. CREATE CORE COMPONENTS
# ============================================================

gnss_monitor = GNSSMonitor()

# We'll initialize this later from the real GPS position.

position_engine = PositionEngine(
    0.0,
    0.0
)

navigation = PRIMENavigation(
    gnss_monitor,
    inference_engine,
    position_engine
)


# ============================================================
# 5. IDENTIFY GPS FIXES
# ============================================================

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

# GPS fix changes
fix_change = (
    lat.ne(lat.shift())
    |
    lon.ne(lon.shift())
)

fix_indices = np.where(
    fix_change
)[0]

print(
    "\nGPS fix transitions found:",
    len(fix_indices)
)


# ============================================================
# 6. LIMIT TO SEQUENCE SAMPLES
# ============================================================

n = min(
    len(X),
    len(fix_indices) - 1
)

print(
    "Samples available for E2E test:",
    n
)


# ============================================================
# 7. START FROM REAL GPS POSITION
# ============================================================

start_idx = fix_indices[0]

start_lat = float(
    lat.iloc[start_idx]
)

start_lon = float(
    lon.iloc[start_idx]
)

position_engine.reset(
    start_lat,
    start_lon
)

print(
    "\nREAL START POSITION"
)

print(
    f"Latitude : {start_lat:.7f}"
)

print(
    f"Longitude: {start_lon:.7f}"
)


# ============================================================
# 8. RUN PRIME DR
# ============================================================

results = []

predicted_north = []
predicted_east = []

actual_north = []
actual_east = []

position_errors = []

current_lat = start_lat
current_lon = start_lon


print("\nRunning end-to-end simulation...")

for i in range(n):

    # ----------------------------------------
    # Current real GPS fix
    # ----------------------------------------

    current_idx = fix_indices[i]

    next_idx = fix_indices[i + 1]

    true_lat = float(
        lat.iloc[next_idx]
    )

    true_lon = float(
        lon.iloc[next_idx]
    )

    # ----------------------------------------
    # Start PRIME outage
    # ----------------------------------------

    result = navigation.update(
        sensor_sequence=X[i],
        satellites=0,
        accuracy=None,
        time_since_update=20.0
    )

    prime_lat = result[
        "position"
    ]["latitude"]

    prime_lon = result[
        "position"
    ]["longitude"]

    # ----------------------------------------
    # Convert position difference
    # to meters
    # ----------------------------------------

    mean_lat = np.radians(
        (true_lat + prime_lat) / 2
    )

    north_error = (
        (prime_lat - true_lat)
        * 111320.0
    )

    east_error = (
        (prime_lon - true_lon)
        * 111320.0
        * np.cos(mean_lat)
    )

    error = np.sqrt(
        north_error ** 2
        +
        east_error ** 2
    )

    # ----------------------------------------
    # Actual displacement from current fix
    # ----------------------------------------

    current_lat_real = float(
        lat.iloc[current_idx]
    )

    current_lon_real = float(
        lon.iloc[current_idx]
    )

    actual_n = (
        (true_lat - current_lat_real)
        * 111320.0
    )

    actual_e = (
        (true_lon - current_lon_real)
        * 111320.0
        * np.cos(
            np.radians(
                current_lat_real
            )
        )
    )

    predicted_north.append(
        prime_lat
    )

    predicted_east.append(
        prime_lon
    )

    actual_north.append(
        true_lat
    )

    actual_east.append(
        true_lon
    )

    position_errors.append(
        error
    )

    results.append({

        "step": i + 1,

        "current_fix_index":
            current_idx,

        "next_fix_index":
            next_idx,

        "actual_latitude":
            true_lat,

        "actual_longitude":
            true_lon,

        "prime_latitude":
            prime_lat,

        "prime_longitude":
            prime_lon,

        "actual_north_displacement_m":
            actual_n,

        "actual_east_displacement_m":
            actual_e,

        "position_error_m":
            error
    })


# ============================================================
# 9. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 10. STATISTICS
# ============================================================

errors = np.array(
    position_errors
)

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


print("\n" + "=" * 70)
print("STEP 45 — E2E RESULTS")
print("=" * 70)

print(
    f"\nSamples tested : {len(errors)}"
)

print(
    f"Mean error     : {mean_error:.3f} m"
)

print(
    f"Median error   : {median_error:.3f} m"
)

print(
    f"P90 error      : {p90_error:.3f} m"
)

print(
    f"Maximum error  : {max_error:.3f} m"
)


# ============================================================
# 11. TRAJECTORY PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.plot(
    actual_east,
    actual_north,
    label="Actual GPS trajectory"
)

plt.plot(
    predicted_east,
    predicted_north,
    label="PRIME trajectory"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "PROJECT PRIME — Actual vs PRIME Trajectory"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# 12. FINAL
# ============================================================

print("\nGenerated:")

print(
    OUTPUT_CSV
)

print(
    OUTPUT_PLOT
)

print("\n" + "=" * 70)
print("STEP 45 COMPLETE")
print("=" * 70)