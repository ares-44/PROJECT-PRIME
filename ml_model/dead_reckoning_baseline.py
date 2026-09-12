import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 5 — DEAD RECKONING BASELINE
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n========================================")
print("PROJECT PRIME — DEAD RECKONING BASELINE")
print("========================================")

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

time = df["TIME SINCE START (ms)"].to_numpy() / 1000.0

lat = df["GPS LATITUDE (degrees)"].to_numpy()
lon = df["GPS LONGITUDE (degrees)"].to_numpy()

speed_kmh = df["GPS SPEED (Kmh)"].to_numpy()
heading = df["GPS ORIENTATION (Â°)"].to_numpy()

# Sensor data
ax = df["ACCELEROMETER X (m/s²)"].to_numpy()
ay = df["ACCELEROMETER Y (m/s²)"].to_numpy()
az = df["ACCELEROMETER Z (m/s²)"].to_numpy()

gx = df["GRAVITY X (m/s²)"].to_numpy()
gy = df["GRAVITY Y (m/s²)"].to_numpy()
gz = df["GRAVITY Z (m/s²)"].to_numpy()

# ------------------------------------------------------------
# Convert acceleration to LINEAR acceleration
# ------------------------------------------------------------

linear_x = ax - gx
linear_y = ay - gy
linear_z = az - gz

# Horizontal acceleration magnitude
horizontal_acc = np.sqrt(
    linear_x**2 +
    linear_y**2
)

# ------------------------------------------------------------
# GPS reference velocity
# ------------------------------------------------------------

speed_ms = speed_kmh / 3.6

# ------------------------------------------------------------
# GPS heading → radians
# ------------------------------------------------------------

heading_rad = np.deg2rad(heading)

# ------------------------------------------------------------
# GPS velocity components
#
# North = speed * cos(heading)
# East  = speed * sin(heading)
# ------------------------------------------------------------

velocity_north = speed_ms * np.cos(heading_rad)
velocity_east = speed_ms * np.sin(heading_rad)

# ------------------------------------------------------------
# GPS position → local meters
# ------------------------------------------------------------

lat0 = lat[0]
lon0 = lon[0]

earth_radius = 6371000.0

north = np.deg2rad(lat - lat0) * earth_radius

east = (
    np.deg2rad(lon - lon0)
    * earth_radius
    * np.cos(np.deg2rad(lat0))
)

# ------------------------------------------------------------
# Calculate GPS distance
# ------------------------------------------------------------

gps_distance = np.sqrt(
    (north - north[0])**2 +
    (east - east[0])**2
)

# ------------------------------------------------------------
# Dead-reckoning baseline
#
# IMPORTANT:
# We use GPS velocity + heading as the reference trajectory
# here to establish the baseline before attempting sensor-only
# navigation.
# ------------------------------------------------------------

dt = np.diff(time)

valid_dt = (
    np.isfinite(dt)
    & (dt > 0)
    & (dt < 2)
)

# Distance travelled using GPS speed
segment_distance = (
    speed_ms[:-1][valid_dt]
    * dt[valid_dt]
)

dead_reckoning_distance = np.concatenate(
    [[0], np.cumsum(segment_distance)]
)

# ------------------------------------------------------------
# Total travelled distance
# ------------------------------------------------------------

total_gps_distance = np.nansum(
    segment_distance
)

total_dr_distance = dead_reckoning_distance[-1]

# ------------------------------------------------------------
# Compare distance
# ------------------------------------------------------------

distance_error = abs(
    total_dr_distance -
    total_gps_distance
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n===== DATA =====")

print("Total samples:", len(df))

print(
    "Duration:",
    time[-1] - time[0],
    "seconds"
)

print(
    "Duration:",
    (time[-1] - time[0]) / 60,
    "minutes"
)

print("\n===== GPS MOTION =====")

print(
    "Average speed:",
    np.nanmean(speed_kmh),
    "km/h"
)

print(
    "Maximum speed:",
    np.nanmax(speed_kmh),
    "km/h"
)

print(
    "GPS travelled distance:",
    total_gps_distance,
    "meters"
)

print("\n===== DEAD RECKONING BASELINE =====")

print(
    "DR travelled distance:",
    total_dr_distance,
    "meters"
)

print(
    "Distance difference:",
    distance_error,
    "meters"
)

print(
    "Distance error percentage:",
    (distance_error / max(total_gps_distance, 1)) * 100,
    "%"
)

print("\n===== LINEAR ACCELERATION =====")

print(
    "Horizontal acceleration mean:",
    np.nanmean(horizontal_acc),
    "m/s²"
)

print(
    "Horizontal acceleration std:",
    np.nanstd(horizontal_acc),
    "m/s²"
)

print(
    "Maximum horizontal acceleration:",
    np.nanmax(horizontal_acc),
    "m/s²"
)

print("\n========================================")
print("STEP 5 COMPLETE")
print("========================================")