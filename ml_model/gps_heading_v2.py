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
# 2. GPS DATA
# --------------------------------------------------

lat = df["GPS LATITUDE (degrees)"].to_numpy()
lon = df["GPS LONGITUDE (degrees)"].to_numpy()
speed = df["GPS SPEED (Kmh)"].to_numpy()

# --------------------------------------------------
# 3. CALCULATE GPS HEADING OVER ~1 SECOND
# --------------------------------------------------

time = df["time_seconds"].to_numpy()

heading_v2 = np.full(len(df), np.nan)
distance_m = np.full(len(df), np.nan)

EARTH_RADIUS = 6371000

for i in range(len(df)):

    target_time = time[i] + 1.0

    # Find the first sample approximately 1 second later
    j = np.searchsorted(time, target_time)

    if j >= len(df):
        continue

    lat1 = np.radians(lat[i])
    lat2 = np.radians(lat[j])

    lon1 = np.radians(lon[i])
    lon2 = np.radians(lon[j])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine distance
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    distance = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))

    distance_m[i] = distance

    # Ignore extremely small GPS movements
    if distance < 2:
        continue

    # Bearing from point 1 → point 2
    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.degrees(np.arctan2(x, y))

    bearing = (bearing + 360) % 360

    heading_v2[i] = bearing

df["GPS_TRAJECTORY_HEADING_V2"] = heading_v2
df["GPS_DISTANCE_1S"] = distance_m

# --------------------------------------------------
# 4. VALID MOVING SAMPLES
# --------------------------------------------------

valid = (
    (df["GPS SPEED (Kmh)"] > 3)
    & (df["GPS_DISTANCE_1S"] >= 2)
    & (df["GPS_TRAJECTORY_HEADING_V2"].notna())
)

moving = df[valid].copy()

print("\n========================================")
print("GPS TRAJECTORY HEADING V2")
print("========================================")

print("\nTotal samples:", len(df))
print("Valid heading samples:", len(moving))

# --------------------------------------------------
# 5. HEADING STATISTICS
# --------------------------------------------------

print("\n===== TRAJECTORY HEADING V2 =====")

print(
    "Mean:",
    moving["GPS_TRAJECTORY_HEADING_V2"].mean()
)

print(
    "Median:",
    moving["GPS_TRAJECTORY_HEADING_V2"].median()
)

print(
    "Minimum:",
    moving["GPS_TRAJECTORY_HEADING_V2"].min()
)

print(
    "Maximum:",
    moving["GPS_TRAJECTORY_HEADING_V2"].max()
)

# --------------------------------------------------
# 6. CIRCULAR MEAN
# --------------------------------------------------

heading_rad = np.radians(
    moving["GPS_TRAJECTORY_HEADING_V2"]
)

circular_mean = np.degrees(
    np.arctan2(
        np.mean(np.sin(heading_rad)),
        np.mean(np.cos(heading_rad))
    )
)

circular_mean = (circular_mean + 360) % 360

print("\nCircular mean heading:", circular_mean)

# --------------------------------------------------
# 7. COMPARE WITH DATASET GPS ORIENTATION
# --------------------------------------------------

gps_orientation = moving["GPS ORIENTATION (Â°)"].to_numpy()

trajectory_heading = moving[
    "GPS_TRAJECTORY_HEADING_V2"
].to_numpy()


def angle_difference(a, b):

    return (a - b + 180) % 360 - 180


heading_error = angle_difference(
    trajectory_heading,
    gps_orientation
)

print("\n===== TRAJECTORY V2 vs GPS ORIENTATION =====")

print(
    "Mean absolute error:",
    np.mean(np.abs(heading_error))
)

print(
    "Median absolute error:",
    np.median(np.abs(heading_error))
)

print(
    "90th percentile error:",
    np.percentile(
        np.abs(heading_error),
        90
    )
)

# --------------------------------------------------
# 8. GPS DISTANCE STATISTICS
# --------------------------------------------------

print("\n===== GPS MOVEMENT =====")

print(
    moving["GPS_DISTANCE_1S"].describe()
)

# --------------------------------------------------
# 9. PLOT HEADING
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    moving["time_seconds"],
    moving["GPS_TRAJECTORY_HEADING_V2"],
    label="GPS Trajectory Heading V2"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")
plt.title("GPS-Derived Vehicle Heading V2")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 10. COMPARE HEADINGS
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    moving["time_seconds"],
    moving["GPS_TRAJECTORY_HEADING_V2"],
    label="Trajectory Heading V2"
)

plt.plot(
    moving["time_seconds"],
    moving["GPS ORIENTATION (Â°)"],
    label="Dataset GPS Orientation"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")
plt.title("Trajectory Heading vs Dataset GPS Orientation")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 11. SPEED VS GPS MOVEMENT
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    moving["time_seconds"],
    moving["GPS SPEED (Kmh)"],
    label="GPS Speed"
)

plt.plot(
    moving["time_seconds"],
    moving["GPS_DISTANCE_1S"],
    label="GPS Distance / ~1 sec"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Value")
plt.title("Vehicle Movement Validation")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 12. SAVE
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\gps_heading_v2_result.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("GPS HEADING V2 COMPLETE")
print("========================================")

print("\nSaved result:")
print(OUTPUT_PATH)