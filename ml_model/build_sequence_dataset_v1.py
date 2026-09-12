import pandas as pd
import numpy as np

print("=" * 70)
print("PROJECT PRIME — SEQUENCE DATASET V1")
print("=" * 70)

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_sequence_dataset_v1.npz"

df = pd.read_csv(INPUT_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))


# =========================================================
# SENSOR COLUMNS
# =========================================================

SENSOR_COLUMNS = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",

    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",

    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)"
]

# Detect magnetometer columns
MAG_COLUMNS = [
    col
    for col in df.columns
    if "MAGNETIC FIELD" in col
]

if len(MAG_COLUMNS) != 3:

    raise ValueError(
        f"Expected 3 magnetometer columns, "
        f"found {len(MAG_COLUMNS)}"
    )

SENSOR_COLUMNS.extend(MAG_COLUMNS)

print("\nSensor columns:", len(SENSOR_COLUMNS))


# =========================================================
# GPS
# =========================================================

TIME = "TIME SINCE START (ms)"
LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"


time = (
    df[TIME].values / 1000.0
)


# =========================================================
# LOCAL GPS COORDINATES
# =========================================================

R = 6371000.0

lat = np.radians(
    df[LAT].values
)

lon = np.radians(
    df[LON].values
)

lat0 = np.mean(lat)

gps_north = lat * R

gps_east = (
    lon *
    R *
    np.cos(lat0)
)


# =========================================================
# GPS FIXES
# =========================================================

gps_changed = (
    df[LAT].diff().fillna(0).ne(0)
    |
    df[LON].diff().fillna(0).ne(0)
)

fix_indices = np.where(
    gps_changed.values
)[0]

print(
    "GPS fixes:",
    len(fix_indices)
)


# =========================================================
# SEQUENCE PARAMETERS
# =========================================================

WINDOW = 20

X = []

y = []

intervals = []

fix_ids = []


# =========================================================
# BUILD SEQUENCES
# =========================================================

for j in range(
    len(fix_indices) - 1
):

    start = fix_indices[j]

    end = fix_indices[j + 1]

    interval = (
        time[end]
        -
        time[start]
    )


    # Keep normal GPS intervals
    if interval < 5 or interval > 15:
        continue


    # Need 20 sensor samples
    if start < WINDOW:
        continue


    # -----------------------------------------------------
    # SENSOR SEQUENCE
    # -----------------------------------------------------

    sequence = df[
        SENSOR_COLUMNS
    ].iloc[
        start - WINDOW + 1 :
        start + 1
    ].values


    if np.isnan(sequence).any():
        continue


    # -----------------------------------------------------
    # GPS DISPLACEMENT TARGET
    # -----------------------------------------------------

    delta_n = (
        gps_north[end]
        -
        gps_north[start]
    )

    delta_e = (
        gps_east[end]
        -
        gps_east[start]
    )


    X.append(sequence)

    y.append([
        delta_n,
        delta_e
    ])

    intervals.append(interval)

    fix_ids.append(start)


# =========================================================
# NUMPY ARRAYS
# =========================================================

X = np.array(X)

y = np.array(y)

intervals = np.array(intervals)

fix_ids = np.array(fix_ids)


# =========================================================
# OUTPUT
# =========================================================

print("\n===== RESULT =====")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Intervals:",
    len(intervals)
)


# =========================================================
# SAVE
# =========================================================

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y,
    intervals=intervals,
    fix_ids=fix_ids
)


print(
    "\nSaved:",
    OUTPUT_PATH
)


print(
    "\n" + "=" * 70
)

print(
    "STEP 26 COMPLETE"
)

print(
    "=" * 70
)