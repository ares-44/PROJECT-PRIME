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
# 2. ACCELEROMETER DATA
# --------------------------------------------------

ax = df["ACCELEROMETER X (m/s²)"]
ay = df["ACCELEROMETER Y (m/s²)"]
az = df["ACCELEROMETER Z (m/s²)"]

# Gravity components
gx = df["GRAVITY X (m/s²)"]
gy = df["GRAVITY Y (m/s²)"]
gz = df["GRAVITY Z (m/s²)"]

# --------------------------------------------------
# 3. REMOVE GRAVITY
# --------------------------------------------------

linear_ax = ax - gx
linear_ay = ay - gy
linear_az = az - gz

df["linear_ax"] = linear_ax
df["linear_ay"] = linear_ay
df["linear_az"] = linear_az

# --------------------------------------------------
# 4. GPS SPEED
# --------------------------------------------------

gps_speed = df["GPS SPEED (Kmh)"]

# Convert km/h → m/s
df["gps_speed_ms"] = gps_speed / 3.6

# Approximate longitudinal acceleration from GPS speed
df["gps_acceleration"] = (
    df["gps_speed_ms"].diff()
    / df["time_seconds"].diff()
)

# Remove unrealistic values caused by tiny time intervals
df.loc[
    df["gps_acceleration"].abs() > 5,
    "gps_acceleration"
] = np.nan

# --------------------------------------------------
# 5. MOVING SAMPLES
# --------------------------------------------------

moving = df[gps_speed > 3].copy()

print("\n========================================")
print("COORDINATE FRAME ANALYSIS")
print("========================================")

print("\nTotal samples:", len(df))
print("Moving samples:", len(moving))

# --------------------------------------------------
# 6. CORRELATION WITH GPS ACCELERATION
# --------------------------------------------------

valid = df[
    ["linear_ax", "linear_ay", "linear_az", "gps_acceleration"]
].dropna()

corr_x = valid["linear_ax"].corr(valid["gps_acceleration"])
corr_y = valid["linear_ay"].corr(valid["gps_acceleration"])
corr_z = valid["linear_az"].corr(valid["gps_acceleration"])

print("\n===== ACCELERATION CORRELATION =====")

print("Phone X vs GPS acceleration:", corr_x)
print("Phone Y vs GPS acceleration:", corr_y)
print("Phone Z vs GPS acceleration:", corr_z)

# --------------------------------------------------
# 7. DETERMINE MOST LIKELY FORWARD AXIS
# --------------------------------------------------

correlations = {
    "X": abs(corr_x),
    "Y": abs(corr_y),
    "Z": abs(corr_z)
}

forward_axis = max(correlations, key=correlations.get)

print("\n===== FORWARD AXIS ESTIMATION =====")
print("Most likely vehicle forward axis:", forward_axis)

if forward_axis == "X":
    sign = np.sign(corr_x)
elif forward_axis == "Y":
    sign = np.sign(corr_y)
else:
    sign = np.sign(corr_z)

print("Estimated direction sign:", sign)

# --------------------------------------------------
# 8. ACCELERATION STATISTICS
# --------------------------------------------------

print("\n===== LINEAR ACCELERATION =====")

print("\nX axis:")
print(df["linear_ax"].describe())

print("\nY axis:")
print(df["linear_ay"].describe())

print("\nZ axis:")
print(df["linear_az"].describe())

# --------------------------------------------------
# 9. PLOT LINEAR ACCELERATION
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["linear_ax"],
    label="Linear Acceleration X"
)

plt.plot(
    df["time_seconds"],
    df["linear_ay"],
    label="Linear Acceleration Y"
)

plt.plot(
    df["time_seconds"],
    df["linear_az"],
    label="Linear Acceleration Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Gravity-Removed Smartphone Acceleration")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 10. GPS ACCELERATION VS SENSOR AXES
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["gps_acceleration"],
    label="GPS-derived acceleration"
)

plt.plot(
    df["time_seconds"],
    df["linear_ax"],
    label="Phone X"
)

plt.plot(
    df["time_seconds"],
    df["linear_ay"],
    label="Phone Y"
)

plt.plot(
    df["time_seconds"],
    df["linear_az"],
    label="Phone Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("GPS Acceleration vs Smartphone Axes")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 11. GYROSCOPE
# --------------------------------------------------

gyro_yaw = df["GYROSCOPE Yaw (rad/s)"]
gyro_pitch = df["GYROSCOPE Pitch (rad/s)"]
gyro_roll = df["GYROSCOPE Roll (rad/s)"]

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    gyro_yaw,
    label="Gyroscope Yaw"
)

plt.plot(
    df["time_seconds"],
    gyro_pitch,
    label="Gyroscope Pitch"
)

plt.plot(
    df["time_seconds"],
    gyro_roll,
    label="Gyroscope Roll"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Vehicle Rotation Information")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 12. SAVE RESULTS
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\coordinate_frame_result.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print("\nCoordinate-frame data saved to:")
print(OUTPUT_PATH)