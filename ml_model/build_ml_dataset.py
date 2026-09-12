import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 8 — BUILD ML TRAINING DATASET
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset.csv"

WINDOW = 20

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n========================================")
print("PROJECT PRIME — ML DATASET BUILDER")
print("========================================")

# ------------------------------------------------------------
# Sensor columns
# ------------------------------------------------------------

sensor_columns = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",
]

# ------------------------------------------------------------
# Create dataframe containing only required data
# ------------------------------------------------------------

data = df[sensor_columns].copy()

# ------------------------------------------------------------
# Add speed as a contextual feature
# ------------------------------------------------------------

data["speed"] = df["GPS SPEED (Kmh)"].to_numpy()

# ------------------------------------------------------------
# Gravity-compensated acceleration
# ------------------------------------------------------------

data["lin_acc_x"] = (
    df["ACCELEROMETER X (m/s²)"]
    - df["GRAVITY X (m/s²)"]
)

data["lin_acc_y"] = (
    df["ACCELEROMETER Y (m/s²)"]
    - df["GRAVITY Y (m/s²)"]
)

data["lin_acc_z"] = (
    df["ACCELEROMETER Z (m/s²)"]
    - df["GRAVITY Z (m/s²)"]
)

# ------------------------------------------------------------
# Magnitudes
# ------------------------------------------------------------

data["acc_magnitude"] = np.sqrt(
    data["lin_acc_x"]**2 +
    data["lin_acc_y"]**2 +
    data["lin_acc_z"]**2
)

data["gyro_magnitude"] = np.sqrt(
    data["GYROSCOPE Yaw (rad/s)"]**2 +
    data["GYROSCOPE Pitch (rad/s)"]**2 +
    data["GYROSCOPE Roll (rad/s)"]**2
)

# ------------------------------------------------------------
# GPS position
# Used ONLY to create training labels.
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
# Future horizon
#
# 20 samples ahead
# ------------------------------------------------------------

future_north = np.roll(north, -WINDOW)
future_east = np.roll(east, -WINDOW)

target_north = (
    future_north - north
)

target_east = (
    future_east - east
)

target_distance = np.sqrt(
    target_north**2 +
    target_east**2
)

# Last WINDOW rows have invalid future targets
target_north[-WINDOW:] = np.nan
target_east[-WINDOW:] = np.nan
target_distance[-WINDOW:] = np.nan

# ------------------------------------------------------------
# Rolling features
# ------------------------------------------------------------

feature_df = pd.DataFrame(index=df.index)

for column in [
    "lin_acc_x",
    "lin_acc_y",
    "lin_acc_z",
    "acc_magnitude",
    "gyro_magnitude",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
    "speed",
]:

    feature_df[column + "_mean"] = (
        data[column]
        .rolling(WINDOW)
        .mean()
    )

    feature_df[column + "_std"] = (
        data[column]
        .rolling(WINDOW)
        .std()
    )

    feature_df[column + "_min"] = (
        data[column]
        .rolling(WINDOW)
        .min()
    )

    feature_df[column + "_max"] = (
        data[column]
        .rolling(WINDOW)
        .max()
    )

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

feature_df["target_north"] = target_north
feature_df["target_east"] = target_east
feature_df["target_distance"] = target_distance

# ------------------------------------------------------------
# Remove invalid rows
# ------------------------------------------------------------

feature_df = feature_df.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_df = feature_df.dropna()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

feature_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n===== ORIGINAL DATA =====")

print(
    "Samples:",
    len(df)
)

print("\n===== ML DATASET =====")

print(
    "Samples:",
    len(feature_df)
)

print(
    "Features:",
    len(feature_df.columns) - 3
)

print(
    "Targets:",
    3
)

print("\n===== TARGET STATISTICS =====")

print(
    "North target mean:",
    feature_df["target_north"].mean(),
    "m"
)

print(
    "East target mean:",
    feature_df["target_east"].mean(),
    "m"
)

print(
    "Distance target mean:",
    feature_df["target_distance"].mean(),
    "m"
)

print(
    "Distance target median:",
    feature_df["target_distance"].median(),
    "m"
)

print(
    "Distance target P90:",
    np.percentile(
        feature_df["target_distance"],
        90
    ),
    "m"
)

print("\n===== SAVED =====")

print(OUTPUT_PATH)

print("\n========================================")
print("STEP 8 COMPLETE")
print("========================================")