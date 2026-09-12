import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 9 — ML DATASET V2
# GNSS-INDEPENDENT FEATURES
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v2.csv"

WINDOW = 20

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n========================================")
print("PROJECT PRIME — ML DATASET V2")
print("========================================")

# ------------------------------------------------------------
# TIME
# ------------------------------------------------------------

time = df["TIME SINCE START (ms)"].to_numpy() / 1000.0

# ------------------------------------------------------------
# RAW SENSOR DATA
# ------------------------------------------------------------

ax = df["ACCELEROMETER X (m/s²)"].to_numpy()
ay = df["ACCELEROMETER Y (m/s²)"].to_numpy()
az = df["ACCELEROMETER Z (m/s²)"].to_numpy()

gx = df["GRAVITY X (m/s²)"].to_numpy()
gy = df["GRAVITY Y (m/s²)"].to_numpy()
gz = df["GRAVITY Z (m/s²)"].to_numpy()

gyro_yaw = df["GYROSCOPE Yaw (rad/s)"].to_numpy()
gyro_pitch = df["GYROSCOPE Pitch (rad/s)"].to_numpy()
gyro_roll = df["GYROSCOPE Roll (rad/s)"].to_numpy()

mag_x = df["MAGNETIC FIELD X (Î¼T)"].to_numpy()
mag_y = df["MAGNETIC FIELD Y (Î¼T)"].to_numpy()
mag_z = df["MAGNETIC FIELD Z (Î¼T)"].to_numpy()

# ------------------------------------------------------------
# GRAVITY-COMPENSATED ACCELERATION
# ------------------------------------------------------------

lin_x = ax - gx
lin_y = ay - gy
lin_z = az - gz

# ------------------------------------------------------------
# MAGNITUDES
# ------------------------------------------------------------

acc_mag = np.sqrt(
    lin_x**2 +
    lin_y**2 +
    lin_z**2
)

gyro_mag = np.sqrt(
    gyro_yaw**2 +
    gyro_pitch**2 +
    gyro_roll**2
)

mag_mag = np.sqrt(
    mag_x**2 +
    mag_y**2 +
    mag_z**2
)

gravity_mag = np.sqrt(
    gx**2 +
    gy**2 +
    gz**2
)

# ------------------------------------------------------------
# GPS → LOCAL NORTH/EAST
#
# GPS is ONLY used for labels.
# ------------------------------------------------------------

lat = df["GPS LATITUDE (degrees)"].to_numpy()
lon = df["GPS LONGITUDE (degrees)"].to_numpy()

R = 6371000.0

lat0 = lat[0]
lon0 = lon[0]

north = (
    np.deg2rad(lat - lat0) * R
)

east = (
    np.deg2rad(lon - lon0)
    * R
    * np.cos(np.deg2rad(lat0))
)

# ------------------------------------------------------------
# FUTURE TARGET
#
# Predict displacement WINDOW samples into the future.
# ------------------------------------------------------------

future_north = np.roll(north, -WINDOW)
future_east = np.roll(east, -WINDOW)

target_north = future_north - north
target_east = future_east - east

target_distance = np.sqrt(
    target_north**2 +
    target_east**2
)

# Last WINDOW samples have no future target.

target_north[-WINDOW:] = np.nan
target_east[-WINDOW:] = np.nan
target_distance[-WINDOW:] = np.nan

# ------------------------------------------------------------
# BUILD FEATURE TABLE
# ------------------------------------------------------------

features = pd.DataFrame(index=df.index)

base_features = {
    "lin_acc_x": lin_x,
    "lin_acc_y": lin_y,
    "lin_acc_z": lin_z,

    "acc_magnitude": acc_mag,

    "gyro_yaw": gyro_yaw,
    "gyro_pitch": gyro_pitch,
    "gyro_roll": gyro_roll,

    "gyro_magnitude": gyro_mag,

    "mag_x": mag_x,
    "mag_y": mag_y,
    "mag_z": mag_z,

    "mag_magnitude": mag_mag,

    "gravity_x": gx,
    "gravity_y": gy,
    "gravity_z": gz,

    "gravity_magnitude": gravity_mag,
}

# ------------------------------------------------------------
# ROLLING STATISTICS
# ------------------------------------------------------------

for name, values in base_features.items():

    series = pd.Series(values)

    features[name + "_mean"] = (
        series.rolling(WINDOW).mean()
    )

    features[name + "_std"] = (
        series.rolling(WINDOW).std()
    )

    features[name + "_min"] = (
        series.rolling(WINDOW).min()
    )

    features[name + "_max"] = (
        series.rolling(WINDOW).max()
    )

# ------------------------------------------------------------
# CHANGE / DYNAMICS FEATURES
# ------------------------------------------------------------

features["acc_change"] = (
    np.sqrt(
        np.diff(lin_x, prepend=lin_x[0])**2 +
        np.diff(lin_y, prepend=lin_y[0])**2 +
        np.diff(lin_z, prepend=lin_z[0])**2
    )
)

features["gyro_change"] = (
    np.sqrt(
        np.diff(gyro_yaw, prepend=gyro_yaw[0])**2 +
        np.diff(gyro_pitch, prepend=gyro_pitch[0])**2 +
        np.diff(gyro_roll, prepend=gyro_roll[0])**2
    )
)

features["mag_change"] = (
    np.sqrt(
        np.diff(mag_x, prepend=mag_x[0])**2 +
        np.diff(mag_y, prepend=mag_y[0])**2 +
        np.diff(mag_z, prepend=mag_z[0])**2
    )
)

# ------------------------------------------------------------
# TIME WINDOW
# ------------------------------------------------------------

features["window_duration"] = (
    pd.Series(time)
    .diff()
    .rolling(WINDOW)
    .sum()
)

# ------------------------------------------------------------
# TARGETS
# ------------------------------------------------------------

features["target_north"] = target_north
features["target_east"] = target_east
features["target_distance"] = target_distance

# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

features = features.dropna()

# ------------------------------------------------------------
# REMOVE VERY SMALL MOVEMENT TARGETS
#
# Keep windows where the future displacement is > 2 m.
# This prevents the model from learning "don't move".
# ------------------------------------------------------------

before_filter = len(features)

features = features[
    features["target_distance"] > 2.0
].copy()

after_filter = len(features)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

features.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

feature_count = len(features.columns) - 3

print("\n===== ORIGINAL DATA =====")

print(
    "Samples:",
    len(df)
)

print("\n===== ML DATASET V2 =====")

print(
    "Samples before movement filter:",
    before_filter
)

print(
    "Samples after movement filter:",
    after_filter
)

print(
    "Features:",
    feature_count
)

print(
    "Targets:",
    3
)

print("\n===== TARGET STATISTICS =====")

print(
    "North displacement mean:",
    features["target_north"].mean(),
    "m"
)

print(
    "East displacement mean:",
    features["target_east"].mean(),
    "m"
)

print(
    "Distance mean:",
    features["target_distance"].mean(),
    "m"
)

print(
    "Distance median:",
    features["target_distance"].median(),
    "m"
)

print(
    "Distance P90:",
    np.percentile(
        features["target_distance"],
        90
    ),
    "m"
)

print(
    "Distance maximum:",
    features["target_distance"].max(),
    "m"
)

print("\n===== GNSS LEAKAGE CHECK =====")

print("GPS speed: NOT USED")
print("GPS heading: NOT USED")
print("GPS position: USED ONLY FOR TARGETS")

print("\n===== SAVED =====")

print(OUTPUT_PATH)

print("\n========================================")
print("STEP 9 COMPLETE")
print("========================================")