import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT PRIME — GPS TRAJECTORY REPAIR")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned-S-S1_repaired.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\n===== ORIGINAL DATA =====")
print("Samples:", len(df))


# ============================================================
# COLUMN NAMES
# ============================================================

LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"
SPEED = "GPS SPEED (Kmh)"
ACCURACY = "GPS ACCURACY (m)"
TIME = "TIME SINCE START (ms)"


# ============================================================
# GPS → LOCAL NORTH/EAST METERS
# ============================================================

R = 6371000.0

lat_rad = np.radians(df[LAT].values)
lon_rad = np.radians(df[LON].values)

lat0 = np.mean(lat_rad)

north = lat_rad * R
east = lon_rad * R * np.cos(lat0)


# ============================================================
# SAMPLE-TO-SAMPLE GPS MOVEMENT
# ============================================================

dn = np.diff(north, prepend=north[0])
de = np.diff(east, prepend=east[0])

gps_step_distance = np.sqrt(dn**2 + de**2)

dt = np.diff(
    df[TIME].values,
    prepend=df[TIME].values[0]
) / 1000.0

dt[dt <= 0] = np.nan

# Speed implied by GPS coordinates
gps_implied_speed = gps_step_distance / dt


# ============================================================
# RECORDED GPS SPEED
# ============================================================

recorded_speed = df[SPEED].values / 3.6


# ============================================================
# GPS QUALITY CHECK
# ============================================================

print("\n===== GPS QUALITY =====")

print(
    "Recorded GPS speed max:",
    np.nanmax(recorded_speed),
    "m/s"
)

print(
    "Coordinate-derived speed max:",
    np.nanmax(gps_implied_speed),
    "m/s"
)

print(
    "Coordinate-derived speed P90:",
    np.nanpercentile(
        gps_implied_speed[np.isfinite(gps_implied_speed)],
        90
    ),
    "m/s"
)


# ============================================================
# OUTLIER DETECTION
# ============================================================

# The vehicle's recorded maximum speed is only ~5.23 m/s.
# Allow generous tolerance for GPS noise.

MAX_PLAUSIBLE_SPEED = 10.0  # m/s (~36 km/h)

speed_outlier = (
    gps_implied_speed > MAX_PLAUSIBLE_SPEED
)


# Very poor GPS accuracy can also indicate unreliable points.
accuracy_outlier = (
    df[ACCURACY].values > 50
)


# Combine both conditions.
bad_gps = speed_outlier | accuracy_outlier


print("\n===== OUTLIER DETECTION =====")

print(
    "Speed outliers:",
    np.sum(speed_outlier)
)

print(
    "Accuracy outliers:",
    np.sum(accuracy_outlier)
)

print(
    "Total GPS points flagged:",
    np.sum(bad_gps)
)

print(
    "Percentage flagged:",
    np.mean(bad_gps) * 100,
    "%"
)


# ============================================================
# STORE ORIGINAL COORDINATES
# ============================================================

df["GPS_LAT_RAW"] = df[LAT]
df["GPS_LON_RAW"] = df[LON]


# ============================================================
# MARK BAD GPS POINTS
# ============================================================

df["GPS_BAD_POINT"] = bad_gps


# ============================================================
# REPAIR GPS USING INTERPOLATION
# ============================================================

lat_clean = df[LAT].copy()
lon_clean = df[LON].copy()

lat_clean[bad_gps] = np.nan
lon_clean[bad_gps] = np.nan


# Interpolate missing GPS points
lat_clean = lat_clean.interpolate(
    method="linear",
    limit_direction="both"
)

lon_clean = lon_clean.interpolate(
    method="linear",
    limit_direction="both"
)


# ============================================================
# LIGHT SMOOTHING
# ============================================================

lat_clean = (
    lat_clean
    .rolling(window=5, center=True, min_periods=1)
    .median()
)

lon_clean = (
    lon_clean
    .rolling(window=5, center=True, min_periods=1)
    .median()
)


# ============================================================
# SAVE REPAIRED GPS
# ============================================================

df["GPS_LAT_REPAIRED"] = lat_clean
df["GPS_LON_REPAIRED"] = lon_clean


# Replace working GPS columns with repaired values
df[LAT] = lat_clean
df[LON] = lon_clean


# ============================================================
# RECHECK TRAJECTORY
# ============================================================

lat_repaired = np.radians(df[LAT].values)
lon_repaired = np.radians(df[LON].values)

north_repaired = lat_repaired * R
east_repaired = lon_repaired * R * np.cos(lat0)

dn_repaired = np.diff(
    north_repaired,
    prepend=north_repaired[0]
)

de_repaired = np.diff(
    east_repaired,
    prepend=east_repaired[0]
)

distance_repaired = np.sqrt(
    dn_repaired**2 +
    de_repaired**2
)

speed_repaired = distance_repaired / dt


# ============================================================
# REPAIRED GPS STATISTICS
# ============================================================

valid_speed = speed_repaired[
    np.isfinite(speed_repaired)
]

print("\n===== REPAIRED GPS =====")

print(
    "Mean step speed:",
    np.mean(valid_speed),
    "m/s"
)

print(
    "Median step speed:",
    np.median(valid_speed),
    "m/s"
)

print(
    "P90 step speed:",
    np.percentile(valid_speed, 90),
    "m/s"
)

print(
    "Maximum step speed:",
    np.max(valid_speed),
    "m/s"
)

print(
    "Points > 10 m/s after repair:",
    np.sum(valid_speed > 10)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n===== SAVED =====")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("STEP 12 COMPLETE")
print("=" * 60)