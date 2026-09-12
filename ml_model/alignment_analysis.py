import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()

# Convert time to seconds
df["time_seconds"] = df["TIME SINCE START (ms)"] / 1000

# --------------------------------------------------
# 1. Phone orientation
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["ORIENTATION (Yaw) (Â°)"],
    label="Yaw"
)

plt.plot(
    df["time_seconds"],
    df["ORIENTATION (Pitch) (Â°)"],
    label="Pitch"
)

plt.plot(
    df["time_seconds"],
    df["ORIENTATION (Roll ) (Â°)"],
    label="Roll"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Orientation (degrees)")
plt.title("Phone Orientation")
plt.legend()
plt.grid()
plt.show()


# --------------------------------------------------
# 2. Gravity components
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["GRAVITY X (m/s²)"],
    label="Gravity X"
)

plt.plot(
    df["time_seconds"],
    df["GRAVITY Y (m/s²)"],
    label="Gravity Y"
)

plt.plot(
    df["time_seconds"],
    df["GRAVITY Z (m/s²)"],
    label="Gravity Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Gravity (m/s²)")
plt.title("Gravity Components")
plt.legend()
plt.grid()
plt.show()


# --------------------------------------------------
# 3. Magnetometer
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time_seconds"],
    df["MAGNETIC FIELD X (Î¼T)"],
    label="Magnetic X"
)

plt.plot(
    df["time_seconds"],
    df["MAGNETIC FIELD Y (Î¼T)"],
    label="Magnetic Y"
)

plt.plot(
    df["time_seconds"],
    df["MAGNETIC FIELD Z (Î¼T)"],
    label="Magnetic Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Magnetic field (µT)")
plt.title("Magnetometer Signals")
plt.legend()
plt.grid()
plt.show()


# --------------------------------------------------
# 4. Calculate average orientation
# --------------------------------------------------

print("\n===== ORIENTATION ANALYSIS =====")

print("\nAverage Yaw:")
print(df["ORIENTATION (Yaw) (Â°)"].mean())

print("\nAverage Pitch:")
print(df["ORIENTATION (Pitch) (Â°)"].mean())

print("\nAverage Roll:")
print(df["ORIENTATION (Roll ) (Â°)"].mean())


# --------------------------------------------------
# 5. Orientation statistics
# --------------------------------------------------

print("\n===== ORIENTATION STATISTICS =====")

print(
    df[
        [
            "ORIENTATION (Yaw) (Â°)",
            "ORIENTATION (Pitch) (Â°)",
            "ORIENTATION (Roll ) (Â°)"
        ]
    ].describe()
)


# --------------------------------------------------
# 6. Gravity statistics
# --------------------------------------------------

print("\n===== GRAVITY STATISTICS =====")

print(
    df[
        [
            "GRAVITY X (m/s²)",
            "GRAVITY Y (m/s²)",
            "GRAVITY Z (m/s²)"
        ]
    ].describe()
)