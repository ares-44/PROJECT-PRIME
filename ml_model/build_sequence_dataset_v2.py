import os
import numpy as np
import pandas as pd

print("=" * 75)
print("PROJECT PRIME — SEQUENCE DATASET V2")
print("=" * 75)

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA = os.path.join(
    BASE,
    "data",
    "cleaned_S-S1.csv"
)

OUT = os.path.join(
    BASE,
    "data",
    "prime_sequence_dataset_v2.npz"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA)

print("\n===== DATA =====")
print("Rows:", len(df))


# ============================================================
# SENSOR COLUMNS
# ============================================================

sensor_columns = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",

    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",

    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
]


# Automatically detect magnetometer columns
mag_columns = [
    c for c in df.columns
    if "MAGNETIC FIELD" in c
]

print("\nMagnetometer columns:")

for c in mag_columns:
    print(" -", c)


if len(mag_columns) != 3:

    raise ValueError(
        "Expected 3 magnetometer columns, "
        f"found {len(mag_columns)}"
    )


sensor_columns.extend(
    mag_columns
)

print(
    "\nTotal sensor channels:",
    len(sensor_columns)
)


# ============================================================
# CHECK REQUIRED GPS COLUMNS
# ============================================================

required_columns = [
    "GPS LATITUDE (degrees)",
    "GPS LONGITUDE (degrees)",
    "TIME SINCE START (ms)"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# FIND GPS FIXES
# ============================================================

lat = df[
    "GPS LATITUDE (degrees)"
].values

lon = df[
    "GPS LONGITUDE (degrees)"
].values

time_ms = df[
    "TIME SINCE START (ms)"
].values


fix_indices = [
    0
]

for i in range(
    1,
    len(df)
):

    if (
        lat[i] != lat[i - 1]
        or
        lon[i] != lon[i - 1]
    ):

        fix_indices.append(i)


print("\n===== GPS FIXES =====")

print(
    "GPS fixes:",
    len(fix_indices)
)


# ============================================================
# PARAMETERS
# ============================================================

SEQUENCE_LENGTH = 60

# Dataset sampling is approximately 10 Hz.
# Therefore:
#
# 60 samples ≈ 6 seconds

print(
    "\nSequence length:",
    SEQUENCE_LENGTH,
    "samples"
)

print(
    "Approx history:",
    SEQUENCE_LENGTH * 0.1,
    "seconds"
)


# ============================================================
# LOCAL GPS CONVERSION
# ============================================================

EARTH_RADIUS = 6371000.0

lat0 = np.deg2rad(
    lat[fix_indices[0]]
)

lon0 = np.deg2rad(
    lon[fix_indices[0]]
)


def gps_to_local(
    latitude,
    longitude
):

    latitude = np.deg2rad(
        latitude
    )

    longitude = np.deg2rad(
        longitude
    )

    north = (
        latitude - lat0
    ) * EARTH_RADIUS

    east = (
        longitude - lon0
    ) * EARTH_RADIUS * np.cos(
        lat0
    )

    return north, east


fix_lat = lat[
    fix_indices
]

fix_lon = lon[
    fix_indices
]


fix_north, fix_east = gps_to_local(
    fix_lat,
    fix_lon
)


# ============================================================
# BUILD SEQUENCES
# ============================================================

X = []

y = []

intervals = []

fix_ids = []


skipped_short_window = 0

skipped_abnormal = 0


for j in range(
    len(fix_indices) - 1
):

    current_fix = fix_indices[j]

    next_fix = fix_indices[j + 1]


    # --------------------------------------------------------
    # GPS interval
    # --------------------------------------------------------

    dt = (
        time_ms[next_fix]
        -
        time_ms[current_fix]
    ) / 1000.0


    # Keep realistic intervals.
    # Normal interval is around 9 seconds.

    if (
        dt < 5.0
        or
        dt > 15.0
    ):

        skipped_abnormal += 1

        continue


    # --------------------------------------------------------
    # Sensor history
    # --------------------------------------------------------

    start = (
        current_fix
        -
        SEQUENCE_LENGTH
    )

    end = current_fix


    if start < 0:

        skipped_short_window += 1

        continue


    sequence = df[
        sensor_columns
    ].iloc[
        start:end
    ].values


    # Must have exact sequence length

    if (
        sequence.shape[0]
        !=
        SEQUENCE_LENGTH
    ):

        skipped_short_window += 1

        continue


    # --------------------------------------------------------
    # Check NaN
    # --------------------------------------------------------

    if np.isnan(
        sequence.astype(float)
    ).any():

        continue


    # --------------------------------------------------------
    # Target displacement
    # --------------------------------------------------------

    target_north = (
        fix_north[j + 1]
        -
        fix_north[j]
    )

    target_east = (
        fix_east[j + 1]
        -
        fix_east[j]
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    X.append(
        sequence.astype(
            np.float32
        )
    )

    y.append(
        [
            target_north,
            target_east
        ]
    )

    intervals.append(
        dt
    )

    fix_ids.append(
        j
    )


# ============================================================
# ARRAYS
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.float32
)

intervals = np.asarray(
    intervals,
    dtype=np.float32
)

fix_ids = np.asarray(
    fix_ids,
    dtype=np.int32
)


# ============================================================
# SUMMARY
# ============================================================

print("\n===== DATASET SUMMARY =====")

print(
    "Valid samples:",
    len(X)
)

print(
    "Skipped abnormal intervals:",
    skipped_abnormal
)

print(
    "Skipped short windows:",
    skipped_short_window
)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# TARGET STATISTICS
# ============================================================

distance = np.sqrt(
    y[:, 0] ** 2
    +
    y[:, 1] ** 2
)


print(
    "\n===== TARGET ====="
)

print(
    "North mean:",
    round(
        np.mean(y[:, 0]),
        4
    ),
    "m"
)

print(
    "East mean:",
    round(
        np.mean(y[:, 1]),
        4
    ),
    "m"
)

print(
    "Distance mean:",
    round(
        np.mean(distance),
        4
    ),
    "m"
)

print(
    "Distance median:",
    round(
        np.median(distance),
        4
    ),
    "m"
)

print(
    "Distance P90:",
    round(
        np.percentile(
            distance,
            90
        ),
        4
    ),
    "m"
)

print(
    "Distance max:",
    round(
        np.max(distance),
        4
    ),
    "m"
)


# ============================================================
# IMPLIED SPEED
# ============================================================

implied_speed = (
    distance
    /
    intervals
)


print(
    "\n===== IMPLIED SPEED ====="
)

print(
    "Mean:",
    round(
        np.mean(implied_speed),
        4
    ),
    "m/s"
)

print(
    "Median:",
    round(
        np.median(implied_speed),
        4
    ),
    "m/s"
)

print(
    "P90:",
    round(
        np.percentile(
            implied_speed,
            90
        ),
        4
    ),
    "m/s"
)

print(
    "Max:",
    round(
        np.max(implied_speed),
        4
    ),
    "m/s"
)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUT,
    X=X,
    y=y,
    intervals=intervals,
    fix_ids=fix_ids,
    sensor_columns=np.array(
        sensor_columns
    )
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
    "STEP 32 DATASET BUILD COMPLETE"
)

print(
    "=" * 75
)