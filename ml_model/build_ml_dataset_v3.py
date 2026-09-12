import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT PRIME — ML DATASET V3")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"


# ============================================================
# SETTINGS
# ============================================================

SENSOR_WINDOW = 20       # ~2 seconds
MIN_FIX_INTERVAL = 5.0   # seconds
MAX_FIX_INTERVAL = 15.0  # seconds


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\n===== ORIGINAL DATA =====")
print("Samples:", len(df))


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

TIME = "TIME SINCE START (ms)"

LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"

SENSOR_COLUMNS = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",

    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",

    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",

    "MAGNETIC FIELD X (Î¼T)",
    "MAGNETIC FIELD Y (Î¼T)",
    "MAGNETIC FIELD Z (Î¼T)"
]


# ============================================================
# CHECK COLUMNS
# ============================================================

required = [
    TIME,
    LAT,
    LON
] + SENSOR_COLUMNS

missing = [
    col for col in required
    if col not in df.columns
]

if missing:

    print("\nERROR — Missing columns:")

    for col in missing:
        print("-", col)

    print("\nAvailable columns:")

    for col in df.columns:
        print("-", col)

    raise SystemExit


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in required:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# FIND ACTUAL GPS FIXES
# ============================================================

gps_changed = (
    df[LAT].diff().fillna(0).ne(0)
    |
    df[LON].diff().fillna(0).ne(0)
)

fix_indices = np.where(
    gps_changed.values
)[0]


print("\n===== GPS FIXES =====")

print(
    "GPS position changes:",
    len(fix_indices)
)


# ============================================================
# GPS → LOCAL NORTH/EAST
# ============================================================

R = 6371000.0

lat_rad = np.radians(
    df[LAT].values
)

lon_rad = np.radians(
    df[LON].values
)

lat0 = np.nanmean(lat_rad)

gps_north = lat_rad * R

gps_east = (
    lon_rad
    * R
    * np.cos(lat0)
)


# ============================================================
# BUILD DATASET
# ============================================================

records = []

skipped_interval = 0
skipped_window = 0
skipped_nan = 0


# ============================================================
# PROCESS GPS FIX INTERVALS
# ============================================================

for j in range(len(fix_indices) - 1):

    current_fix = fix_indices[j]
    next_fix = fix_indices[j + 1]

    # --------------------------------------------------------
    # TIME BETWEEN GPS FIXES
    # --------------------------------------------------------

    t1 = (
        df[TIME].iloc[current_fix]
        / 1000.0
    )

    t2 = (
        df[TIME].iloc[next_fix]
        / 1000.0
    )

    interval = t2 - t1

    # Ignore abnormal GPS gaps
    if (
        interval < MIN_FIX_INTERVAL
        or
        interval > MAX_FIX_INTERVAL
    ):
        skipped_interval += 1
        continue


    # --------------------------------------------------------
    # GPS TARGET
    # --------------------------------------------------------

    target_north = (
        gps_north[next_fix]
        - gps_north[current_fix]
    )

    target_east = (
        gps_east[next_fix]
        - gps_east[current_fix]
    )

    target_distance = np.sqrt(
        target_north ** 2
        +
        target_east ** 2
    )


    # --------------------------------------------------------
    # LAST 2 SECONDS OF SENSOR HISTORY
    #
    # The model sees sensor information BEFORE
    # the current GPS fix.
    # --------------------------------------------------------

    window_end = current_fix

    window_start = (
        window_end - SENSOR_WINDOW + 1
    )

    if window_start < 0:

        skipped_window += 1
        continue


    sensor_window = df.iloc[
        window_start:
        window_end + 1
    ][SENSOR_COLUMNS]


    # --------------------------------------------------------
    # CHECK SENSOR DATA
    # --------------------------------------------------------

    if sensor_window.isna().any().any():

        skipped_nan += 1
        continue


    # --------------------------------------------------------
    # SENSOR FEATURES
    # --------------------------------------------------------

    features = {}

    for col in SENSOR_COLUMNS:

        values = sensor_window[col].values

        features[col + "_mean"] = np.mean(values)
        features[col + "_std"] = np.std(values)
        features[col + "_min"] = np.min(values)
        features[col + "_max"] = np.max(values)


    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    features["target_north"] = target_north
    features["target_east"] = target_east
    features["target_distance"] = target_distance

    # Metadata — NOT ML features
    features["fix_interval_seconds"] = interval
    features["gps_fix_index"] = next_fix

    records.append(features)


# ============================================================
# CREATE DATAFRAME
# ============================================================

ml_df = pd.DataFrame(records)


print("\n===== DATASET V3 =====")

print(
    "Valid samples:",
    len(ml_df)
)

feature_columns = [
    c for c in ml_df.columns
    if c not in [
        "target_north",
        "target_east",
        "target_distance",
        "fix_interval_seconds",
        "gps_fix_index"
    ]
]

print(
    "Features:",
    len(feature_columns)
)

print(
    "Skipped abnormal GPS intervals:",
    skipped_interval
)

print(
    "Skipped sensor windows:",
    skipped_window
)

print(
    "Skipped NaN windows:",
    skipped_nan
)


# ============================================================
# TARGET STATISTICS
# ============================================================

if len(ml_df) > 0:

    print("\n===== TARGET STATISTICS =====")

    print("\n----- North displacement -----")

    print(
        "Mean:",
        ml_df["target_north"].mean(),
        "m"
    )

    print(
        "Median:",
        ml_df["target_north"].median(),
        "m"
    )

    print(
        "P90 absolute:",
        ml_df["target_north"].abs().quantile(0.90),
        "m"
    )


    print("\n----- East displacement -----")

    print(
        "Mean:",
        ml_df["target_east"].mean(),
        "m"
    )

    print(
        "Median:",
        ml_df["target_east"].median(),
        "m"
    )

    print(
        "P90 absolute:",
        ml_df["target_east"].abs().quantile(0.90),
        "m"
    )


    print("\n----- Distance -----")

    print(
        "Mean:",
        ml_df["target_distance"].mean(),
        "m"
    )

    print(
        "Median:",
        ml_df["target_distance"].median(),
        "m"
    )

    print(
        "P90:",
        ml_df["target_distance"].quantile(0.90),
        "m"
    )

    print(
        "Maximum:",
        ml_df["target_distance"].max(),
        "m"
    )


    # ========================================================
    # TARGET SPEED SANITY
    # ========================================================

    implied_speed = (
        ml_df["target_distance"]
        /
        ml_df["fix_interval_seconds"]
    )

    print("\n===== TARGET SPEED SANITY =====")

    print(
        "Mean:",
        implied_speed.mean(),
        "m/s"
    )

    print(
        "Median:",
        implied_speed.median(),
        "m/s"
    )

    print(
        "P90:",
        implied_speed.quantile(0.90),
        "m/s"
    )

    print(
        "Maximum:",
        implied_speed.max(),
        "m/s"
    )

    print(
        "Maximum km/h:",
        implied_speed.max() * 3.6
    )


# ============================================================
# GNSS LEAKAGE CHECK
# ============================================================

print("\n===== GNSS LEAKAGE CHECK =====")

for forbidden in [
    "GPS SPEED",
    "GPS ORIENTATION",
    "GPS LATITUDE",
    "GPS LONGITUDE"
]:

    leakage = [
        c for c in feature_columns
        if forbidden in c
    ]

    print(
        forbidden + ":",
        "FOUND" if leakage else "NOT USED"
    )


# ============================================================
# SAVE
# ============================================================

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== SAVED =====")

print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("STEP 15 COMPLETE")
print("=" * 60)