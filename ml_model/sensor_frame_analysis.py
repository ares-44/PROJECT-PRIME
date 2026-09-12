import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 4A — SENSOR FRAME ANALYSIS
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# Columns
# ------------------------------------------------------------

time = df["TIME SINCE START (ms)"].to_numpy()

speed_kmh = df["GPS SPEED (Kmh)"].to_numpy()

ax = df["ACCELEROMETER X (m/s²)"].to_numpy()
ay = df["ACCELEROMETER Y (m/s²)"].to_numpy()
az = df["ACCELEROMETER Z (m/s²)"].to_numpy()

gx = df["GRAVITY X (m/s²)"].to_numpy()
gy = df["GRAVITY Y (m/s²)"].to_numpy()
gz = df["GRAVITY Z (m/s²)"].to_numpy()

# ------------------------------------------------------------
# Convert speed to m/s
# ------------------------------------------------------------

speed_ms = speed_kmh / 3.6

# ------------------------------------------------------------
# Calculate GPS-derived acceleration
#
# a = dv / dt
# ------------------------------------------------------------

dt = np.gradient(time / 1000.0)

dt[dt <= 0] = np.nan

gps_acceleration = np.gradient(speed_ms) / dt

# ------------------------------------------------------------
# Remove invalid values
# ------------------------------------------------------------

valid = (
    np.isfinite(gps_acceleration)
    & np.isfinite(ax)
    & np.isfinite(ay)
    & np.isfinite(az)
    & (speed_kmh > 3)
)

gps_a = gps_acceleration[valid]

acc_x = ax[valid]
acc_y = ay[valid]
acc_z = az[valid]

# ------------------------------------------------------------
# Correlation
# ------------------------------------------------------------

corr_x = np.corrcoef(gps_a, acc_x)[0, 1]
corr_y = np.corrcoef(gps_a, acc_y)[0, 1]
corr_z = np.corrcoef(gps_a, acc_z)[0, 1]

print("\n========================================")
print("PROJECT PRIME — SENSOR FRAME ANALYSIS")
print("========================================")

print("\nTotal samples:", len(df))
print("Valid moving samples:", len(gps_a))

print("\n===== GPS LONGITUDINAL ACCELERATION =====")

print("Mean:", np.mean(gps_a))
print("Std:", np.std(gps_a))
print("Min:", np.min(gps_a))
print("Max:", np.max(gps_a))

print("\n===== CORRELATION WITH GPS ACCELERATION =====")

print("Accelerometer X:", corr_x)
print("Accelerometer Y:", corr_y)
print("Accelerometer Z:", corr_z)

# ------------------------------------------------------------
# Determine strongest axis
# ------------------------------------------------------------

correlations = {
    "X": abs(corr_x),
    "Y": abs(corr_y),
    "Z": abs(corr_z)
}

best_axis = max(correlations, key=correlations.get)

print("\n===== RESULT =====")

print("Strongest acceleration axis:", best_axis)
print(
    "Absolute correlation:",
    correlations[best_axis]
)

# ------------------------------------------------------------
# Gravity direction
# ------------------------------------------------------------

gravity_magnitude = np.sqrt(
    gx**2 + gy**2 + gz**2
)

print("\n===== GRAVITY =====")

print("Mean gravity magnitude:",
      np.nanmean(gravity_magnitude))

print("Mean Gravity X:",
      np.nanmean(gx))

print("Mean Gravity Y:",
      np.nanmean(gy))

print("Mean Gravity Z:",
      np.nanmean(gz))

print("\n========================================")
print("STEP 4A COMPLETE")
print("========================================")