import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

# -----------------------------
# Convert time from milliseconds
# to seconds
# -----------------------------
df["time_seconds"] = df["TIME SINCE START (ms)"] / 1000

# -----------------------------
# Acceleration magnitude
# -----------------------------
df["acceleration_magnitude"] = (
    df["ACCELEROMETER X (m/s²)"] ** 2
    + df["ACCELEROMETER Y (m/s²)"] ** 2
    + df["ACCELEROMETER Z (m/s²)"] ** 2
) ** 0.5

# -----------------------------
# Plot GPS speed
# -----------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["GPS SPEED (Kmh)"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("GPS Speed (km/h)")
plt.title("Vehicle Speed")
plt.grid()
plt.show()

# -----------------------------
# Plot acceleration magnitude
# -----------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["acceleration_magnitude"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration Magnitude (m/s²)")
plt.title("IMU Acceleration")
plt.grid()
plt.show()

# -----------------------------
# Plot gyroscope
# -----------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["GYROSCOPE Yaw (rad/s)"],
    label="Yaw"
)

plt.plot(
    df["time_seconds"],
    df["GYROSCOPE Pitch (rad/s)"],
    label="Pitch"
)

plt.plot(
    df["time_seconds"],
    df["GYROSCOPE Roll (rad/s)"],
    label="Roll"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Gyroscope Signals")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Print analysis
# -----------------------------
print("\n===== IMU ANALYSIS =====")

print("Total samples:", len(df))

print(
    "Duration:",
    df["time_seconds"].iloc[-1],
    "seconds"
)

print("\nGPS SPEED statistics:")
print(df["GPS SPEED (Kmh)"].describe())

print("\nAcceleration magnitude statistics:")
print(df["acceleration_magnitude"].describe())