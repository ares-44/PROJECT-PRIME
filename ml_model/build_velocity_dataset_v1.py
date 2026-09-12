import pandas as pd
import numpy as np

print("=" * 70)
print("PROJECT PRIME — VELOCITY DATASET V1")
print("=" * 70)

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_velocity_dataset_v1.csv"

df = pd.read_csv(INPUT_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))


# =========================================================
# COLUMNS
# =========================================================

TIME = "TIME SINCE START (ms)"
LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"


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


# =========================================================
# AUTOMATIC MAGNETOMETER DETECTION
# =========================================================

MAG_COLUMNS = [
    col for col in df.columns
    if "MAGNETIC FIELD" in col
]

print("\n===== MAGNETOMETER COLUMNS =====")

for col in MAG_COLUMNS:
    print(col)

if len(MAG_COLUMNS) != 3:
    raise ValueError(
        f"Expected 3 magnetic field columns, "
        f"but found {len(MAG_COLUMNS)}"
    )

SENSOR_COLUMNS.extend(MAG_COLUMNS)

print("\nTotal sensor columns:", len(SENSOR_COLUMNS))


# =========================================================
# CHECK SENSOR COLUMNS
# =========================================================

missing_columns = [
    col for col in SENSOR_COLUMNS
    if col not in df.columns
]

if missing_columns:

    print("\nERROR — Missing columns:")

    for col in missing_columns:
        print(col)

    raise ValueError(
        "Some required sensor columns are missing."
    )


# =========================================================
# TIME
# =========================================================

time = df[TIME].values / 1000.0


# =========================================================
# GPS LOCAL COORDINATES
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

gps_east = lon * R * np.cos(lat0)


# =========================================================
# GPS FIX DETECTION
# =========================================================

gps_changed = (
    df[LAT].diff().fillna(0).ne(0)
    |
    df[LON].diff().fillna(0).ne(0)
)

fix_indices = np.where(
    gps_changed.values
)[0]

print("\n===== GPS FIXES =====")
print("GPS fixes:", len(fix_indices))


# =========================================================
# BUILD VELOCITY DATASET
# =========================================================

WINDOW = 20

rows = []

skipped_interval = 0
skipped_window = 0


for j in range(
    len(fix_indices) - 1
):

    start = fix_indices[j]

    end = fix_indices[j + 1]


    # -----------------------------------------------------
    # GPS FIX INTERVAL
    # -----------------------------------------------------

    interval = (
        time[end]
        -
        time[start]
    )


    # Keep normal GPS intervals
    if interval < 5 or interval > 15:

        skipped_interval += 1

        continue


    # -----------------------------------------------------
    # SENSOR HISTORY
    # -----------------------------------------------------

    if start < WINDOW:

        skipped_window += 1

        continue


    # -----------------------------------------------------
    # GPS DISPLACEMENT
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


    # -----------------------------------------------------
    # VELOCITY TARGET
    # -----------------------------------------------------

    velocity_n = (
        delta_n / interval
    )

    velocity_e = (
        delta_e / interval
    )

    velocity_magnitude = np.sqrt(
        velocity_n ** 2
        +
        velocity_e ** 2
    )


    # -----------------------------------------------------
    # SENSOR WINDOW
    # -----------------------------------------------------

    window = df[
        SENSOR_COLUMNS
    ].iloc[
        start - WINDOW + 1 :
        start + 1
    ]


    # -----------------------------------------------------
    # NaN CHECK
    # -----------------------------------------------------

    if window.isna().any().any():

        skipped_window += 1

        continue


    # -----------------------------------------------------
    # FEATURE EXTRACTION
    # -----------------------------------------------------

    features = {}


    for sensor_index, col in enumerate(
        SENSOR_COLUMNS
    ):

        values = window[col].values


        # Create clean unique feature name
        prefix = f"sensor_{sensor_index}"


        features[
            prefix + "_mean"
        ] = np.mean(values)


        features[
            prefix + "_std"
        ] = np.std(values)


        features[
            prefix + "_min"
        ] = np.min(values)


        features[
            prefix + "_max"
        ] = np.max(values)


    # -----------------------------------------------------
    # TARGETS
    # -----------------------------------------------------

    features[
        "target_velocity_north"
    ] = velocity_n


    features[
        "target_velocity_east"
    ] = velocity_e


    features[
        "target_velocity"
    ] = velocity_magnitude


    # Metadata
    features[
        "interval_seconds"
    ] = interval


    features[
        "gps_fix_index"
    ] = start


    rows.append(features)


# =========================================================
# CREATE DATAFRAME
# =========================================================

result = pd.DataFrame(rows)


# =========================================================
# RESULTS
# =========================================================

print("\n===== RESULT =====")

print(
    "Valid velocity samples:",
    len(result)
)

print(
    "Skipped abnormal intervals:",
    skipped_interval
)

print(
    "Skipped sensor windows:",
    skipped_window
)


# =========================================================
# VELOCITY STATISTICS
# =========================================================

if len(result) > 0:

    velocity = result[
        "target_velocity"
    ]


    print(
        "\n===== VELOCITY STATISTICS ====="
    )


    print(
        "Mean:",
        velocity.mean(),
        "m/s"
    )


    print(
        "Median:",
        velocity.median(),
        "m/s"
    )


    print(
        "P90:",
        velocity.quantile(0.90),
        "m/s"
    )


    print(
        "Maximum:",
        velocity.max(),
        "m/s"
    )


    # -----------------------------------------------------
    # NORTH VELOCITY
    # -----------------------------------------------------

    print(
        "\n===== NORTH VELOCITY ====="
    )

    print(
        "Mean:",
        result[
            "target_velocity_north"
        ].mean(),
        "m/s"
    )


    # -----------------------------------------------------
    # EAST VELOCITY
    # -----------------------------------------------------

    print(
        "\n===== EAST VELOCITY ====="
    )

    print(
        "Mean:",
        result[
            "target_velocity_east"
        ].mean(),
        "m/s"
    )


    # =====================================================
    # SAVE
    # =====================================================

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "\n===== SAVED ====="
    )

    print(
        OUTPUT_PATH
    )


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 23 COMPLETE"
)

print(
    "=" * 70
)