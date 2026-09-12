import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# --------------------------------------------------
# 1. TIME
# --------------------------------------------------

df["time_seconds"] = df["TIME SINCE START (ms)"] / 1000

# --------------------------------------------------
# 2. GPS COORDINATES
# --------------------------------------------------

lat = np.radians(df["GPS LATITUDE (degrees)"].to_numpy())
lon = np.radians(df["GPS LONGITUDE (degrees)"].to_numpy())

# --------------------------------------------------
# 3. CALCULATE GPS TRAJECTORY HEADING
# --------------------------------------------------
# Heading:
# 0°   = North
# 90°  = East
# 180° = South
# 270° = West

lat1 = lat[:-1]
lat2 = lat[1:]

lon1 = lon[:-1]
lon2 = lon[1:]

delta_lon = lon2 - lon1

x = np.sin(delta_lon) * np.cos(lat2)

y = (
    np.cos(lat1) * np.sin(lat2)
    - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
)

heading = np.degrees(np.arctan2(x, y))

heading = (heading + 360) % 360

# First sample has no previous point
df["GPS_TRAJECTORY_HEADING"] = np.nan
df.loc[1:, "GPS_TRAJECTORY_HEADING"] = heading

# --------------------------------------------------
# 4. REMOVE GPS JUMPS
# --------------------------------------------------

# Only calculate trajectory heading when:
# - vehicle is moving
# - GPS speed is reasonable

moving = df["GPS SPEED (Kmh)"] > 3

df.loc[~moving, "GPS_TRAJECTORY_HEADING"] = np.nan

# --------------------------------------------------
# 5. SMOOTH HEADING
# --------------------------------------------------

# Convert heading to radians before smoothing
heading_rad = np.radians(df["GPS_TRAJECTORY_HEADING"])

sin_heading = np.sin(heading_rad)
cos_heading = np.cos(heading_rad)

sin_smooth = (
    pd.Series(sin_heading)
    .rolling(window=10, center=True)
    .mean()
)

cos_smooth = (
    pd.Series(cos_heading)
    .rolling(window=10, center=True)
    .mean()
)

smooth_heading = np.degrees(
    np.arctan2(sin_smooth, cos_smooth)
)

smooth_heading = (smooth_heading + 360) % 360

df["GPS_SMOOTH_HEADING"] = smooth_heading

# --------------------------------------------------
# 6. COMPARE WITH DATASET GPS ORIENTATION
# --------------------------------------------------

gps_orientation = df["GPS ORIENTATION (Â°)"]

def angle_difference(a, b):
    return (a - b + 180) % 360 - 180

valid = df[
    ["GPS_TRAJECTORY_HEADING", "GPS_SMOOTH_HEADING",
     "GPS ORIENTATION (Â°)", "GPS SPEED (Kmh)"]
].dropna()

trajectory_error = angle_difference(
    valid["GPS_TRAJECTORY_HEADING"].to_numpy(),
    valid["GPS ORIENTATION (Â°)"].to_numpy()
)

smooth_error = angle_difference(
    valid["GPS_SMOOTH_HEADING"].to_numpy(),
    valid["GPS ORIENTATION (Â°)"].to_numpy()
)

# --------------------------------------------------
# 7. RESULTS
# --------------------------------------------------

print("\n========================================")
print("GPS TRAJECTORY HEADING ANALYSIS")
print("========================================")

print("\nTotal samples:", len(df))
print("Moving samples:", moving.sum())

print("\n===== TRAJECTORY HEADING =====")

print(
    "Mean:",
    np.nanmean(df["GPS_TRAJECTORY_HEADING"])
)

print(
    "Median:",
    np.nanmedian(df["GPS_TRAJECTORY_HEADING"])
)

print(
    "Minimum:",
    np.nanmin(df["GPS_TRAJECTORY_HEADING"])
)

print(
    "Maximum:",
    np.nanmax(df["GPS_TRAJECTORY_HEADING"])
)

print("\n===== DATASET GPS ORIENTATION =====")

print(
    "Mean:",
    gps_orientation.mean()
)

print(
    "Median:",
    gps_orientation.median()
)

print(
    "Minimum:",
    gps_orientation.min()
)

print(
    "Maximum:",
    gps_orientation.max()
)

print("\n===== TRAJECTORY vs GPS ORIENTATION =====")

print(
    "Mean absolute error:",
    np.mean(np.abs(trajectory_error))
)

print(
    "Median absolute error:",
    np.median(np.abs(trajectory_error))
)

print(
    "90th percentile error:",
    np.percentile(np.abs(trajectory_error), 90)
)

print("\n===== SMOOTH TRAJECTORY vs GPS ORIENTATION =====")

print(
    "Mean absolute error:",
    np.mean(np.abs(smooth_error))
)

print(
    "Median absolute error:",
    np.median(np.abs(smooth_error))
)

print(
    "90th percentile error:",
    np.percentile(np.abs(smooth_error), 90)
)

# --------------------------------------------------
# 8. PLOT TRAJECTORY
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["GPS_TRAJECTORY_HEADING"],
    label="GPS Trajectory Heading"
)

plt.plot(
    df["time_seconds"],
    df["GPS_SMOOTH_HEADING"],
    label="Smoothed Trajectory Heading"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")
plt.title("Vehicle Heading from GPS Trajectory")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 9. COMPARE TRAJECTORY WITH GPS ORIENTATION
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["GPS_TRAJECTORY_HEADING"],
    label="Trajectory Heading"
)

plt.plot(
    df["time_seconds"],
    df["GPS ORIENTATION (Â°)"],
    label="Dataset GPS Orientation"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")
plt.title("GPS Trajectory Heading vs GPS Orientation")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 10. HEADING VS SPEED
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["GPS SPEED (Kmh)"],
    label="GPS Speed"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Speed (km/h)")
plt.title("Vehicle Speed")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\gps_heading_result.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("GPS HEADING ANALYSIS COMPLETE")
print("========================================")

print("\nSaved result:")
print(OUTPUT_PATH)