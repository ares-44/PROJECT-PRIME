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
# 2. SENSOR DATA
# --------------------------------------------------

df["linear_ax"] = (
    df["ACCELEROMETER X (m/s²)"]
    - df["GRAVITY X (m/s²)"]
)

df["linear_ay"] = (
    df["ACCELEROMETER Y (m/s²)"]
    - df["GRAVITY Y (m/s²)"]
)

df["linear_az"] = (
    df["ACCELEROMETER Z (m/s²)"]
    - df["GRAVITY Z (m/s²)"]
)

# --------------------------------------------------
# 3. GPS SPEED
# --------------------------------------------------

df["gps_speed_ms"] = (
    df["GPS SPEED (Kmh)"] / 3.6
)

# --------------------------------------------------
# 4. SMOOTH GPS SPEED
# --------------------------------------------------

# GPS speed can contain repeated/noisy values.
# Smooth it over approximately 1 second.

df["gps_speed_smooth"] = (
    df["gps_speed_ms"]
    .rolling(
        window=10,
        center=True,
        min_periods=1
    )
    .mean()
)

# --------------------------------------------------
# 5. GPS-DERIVED ACCELERATION
# --------------------------------------------------

dt = df["time_seconds"].diff()

df["gps_acceleration"] = (
    df["gps_speed_smooth"].diff() / dt
)

# Remove unrealistic acceleration values
df.loc[
    df["gps_acceleration"].abs() > 5,
    "gps_acceleration"
] = np.nan

# --------------------------------------------------
# 6. SMOOTH IMU ACCELERATION
# --------------------------------------------------

df["ax_smooth"] = (
    df["linear_ax"]
    .rolling(
        window=5,
        center=True,
        min_periods=1
    )
    .mean()
)

df["ay_smooth"] = (
    df["linear_ay"]
    .rolling(
        window=5,
        center=True,
        min_periods=1
    )
    .mean()
)

df["az_smooth"] = (
    df["linear_az"]
    .rolling(
        window=5,
        center=True,
        min_periods=1
    )
    .mean()
)

# --------------------------------------------------
# 7. VALID DATA
# --------------------------------------------------

valid = df[
    (df["GPS SPEED (Kmh)"] > 3)
    & df["gps_acceleration"].notna()
].copy()

print("\n========================================")
print("FORWARD AXIS ANALYSIS")
print("========================================")

print("\nTotal samples:", len(df))
print("Valid samples:", len(valid))

# --------------------------------------------------
# 8. CORRELATION
# --------------------------------------------------

corr_x = valid["ax_smooth"].corr(
    valid["gps_acceleration"]
)

corr_y = valid["ay_smooth"].corr(
    valid["gps_acceleration"]
)

corr_z = valid["az_smooth"].corr(
    valid["gps_acceleration"]
)

print("\n===== IMU vs GPS ACCELERATION =====")

print(
    "Phone X correlation:",
    corr_x
)

print(
    "Phone Y correlation:",
    corr_y
)

print(
    "Phone Z correlation:",
    corr_z
)

# --------------------------------------------------
# 9. ABSOLUTE CORRELATION
# --------------------------------------------------

correlations = {
    "X": abs(corr_x),
    "Y": abs(corr_y),
    "Z": abs(corr_z)
}

forward_axis = max(
    correlations,
    key=correlations.get
)

print("\n===== FORWARD AXIS ESTIMATION =====")

print(
    "Most correlated phone axis:",
    forward_axis
)

if forward_axis == "X":
    direction_sign = np.sign(corr_x)

elif forward_axis == "Y":
    direction_sign = np.sign(corr_y)

else:
    direction_sign = np.sign(corr_z)

print(
    "Estimated direction sign:",
    direction_sign
)

# --------------------------------------------------
# 10. PRINT CORRELATION RANKING
# --------------------------------------------------

print("\n===== CORRELATION RANKING =====")

sorted_corr = sorted(
    correlations.items(),
    key=lambda x: x[1],
    reverse=True
)

for axis, value in sorted_corr:

    print(
        axis,
        "absolute correlation:",
        value
    )

# --------------------------------------------------
# 11. ACCELERATION STATISTICS
# --------------------------------------------------

print("\n===== VALID ACCELERATION =====")

print("\nGPS acceleration:")
print(
    valid["gps_acceleration"].describe()
)

print("\nPhone X:")
print(
    valid["ax_smooth"].describe()
)

print("\nPhone Y:")
print(
    valid["ay_smooth"].describe()
)

print("\nPhone Z:")
print(
    valid["az_smooth"].describe()
)

# --------------------------------------------------
# 12. PLOT GPS ACCELERATION + PHONE X
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    valid["time_seconds"],
    valid["gps_acceleration"],
    label="GPS-derived acceleration"
)

plt.plot(
    valid["time_seconds"],
    valid["ax_smooth"],
    label="Phone X"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("GPS Acceleration vs Phone X")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 13. PLOT GPS ACCELERATION + PHONE Y
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    valid["time_seconds"],
    valid["gps_acceleration"],
    label="GPS-derived acceleration"
)

plt.plot(
    valid["time_seconds"],
    valid["ay_smooth"],
    label="Phone Y"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("GPS Acceleration vs Phone Y")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 14. PLOT GPS ACCELERATION + PHONE Z
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    valid["time_seconds"],
    valid["gps_acceleration"],
    label="GPS-derived acceleration"
)

plt.plot(
    valid["time_seconds"],
    valid["az_smooth"],
    label="Phone Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("GPS Acceleration vs Phone Z")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 15. SAVE RESULTS
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\forward_axis_result.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("FORWARD AXIS ANALYSIS COMPLETE")
print("========================================")

print("\nSaved result:")
print(OUTPUT_PATH)