import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 4E — GYROSCOPE FRAME ANALYSIS
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

time = df["TIME SINCE START (ms)"].to_numpy() / 1000.0

speed = df["GPS SPEED (Kmh)"].to_numpy()

gps_heading = df["GPS ORIENTATION (Â°)"].to_numpy()

gyro_yaw = df["GYROSCOPE Yaw (rad/s)"].to_numpy()
gyro_pitch = df["GYROSCOPE Pitch (rad/s)"].to_numpy()
gyro_roll = df["GYROSCOPE Roll (rad/s)"].to_numpy()

# ------------------------------------------------------------
# Moving samples
# ------------------------------------------------------------

moving = (
    (speed > 5)
    & np.isfinite(gps_heading)
    & np.isfinite(gyro_yaw)
    & np.isfinite(gyro_pitch)
    & np.isfinite(gyro_roll)
    & np.isfinite(time)
)

gps_heading = gps_heading[moving]

gy = gyro_yaw[moving]
gp = gyro_pitch[moving]
gr = gyro_roll[moving]

time = time[moving]

# ------------------------------------------------------------
# Unwrap GPS heading
# ------------------------------------------------------------

gps_rad = np.unwrap(
    np.deg2rad(gps_heading)
)

gps_unwrapped = np.rad2deg(gps_rad)

# ------------------------------------------------------------
# GPS heading rate
#
# degrees/sec
# ------------------------------------------------------------

dt = np.diff(time)

heading_change = np.diff(gps_unwrapped)

valid_dt = (
    np.isfinite(dt)
    & (dt > 0)
    & (dt < 2)
)

heading_rate = (
    heading_change[valid_dt]
    / dt[valid_dt]
)

# ------------------------------------------------------------
# Match gyro samples
# ------------------------------------------------------------

gy = gy[1:][valid_dt]
gp = gp[1:][valid_dt]
gr = gr[1:][valid_dt]

# Convert gyro from rad/s to deg/s
gy_deg = np.rad2deg(gy)
gp_deg = np.rad2deg(gp)
gr_deg = np.rad2deg(gr)

# ------------------------------------------------------------
# Remove extreme GPS heading-rate noise
# ------------------------------------------------------------

valid = (
    np.isfinite(heading_rate)
    & np.isfinite(gy_deg)
    & np.isfinite(gp_deg)
    & np.isfinite(gr_deg)
    & (np.abs(heading_rate) < 50)
)

gps_rate = heading_rate[valid]

gy_deg = gy_deg[valid]
gp_deg = gp_deg[valid]
gr_deg = gr_deg[valid]

# ------------------------------------------------------------
# Correlations
# ------------------------------------------------------------

corr_yaw = np.corrcoef(
    gps_rate,
    gy_deg
)[0, 1]

corr_pitch = np.corrcoef(
    gps_rate,
    gp_deg
)[0, 1]

corr_roll = np.corrcoef(
    gps_rate,
    gr_deg
)[0, 1]

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\n========================================")
print("PROJECT PRIME — GYROSCOPE FRAME ANALYSIS")
print("========================================")

print("Moving samples:", np.sum(moving))
print("Valid turning samples:", len(gps_rate))

print("\n===== GPS TURN RATE =====")

print("Mean:", np.mean(gps_rate))
print("Std:", np.std(gps_rate))
print("Min:", np.min(gps_rate))
print("Max:", np.max(gps_rate))

print("\n===== GYROSCOPE STATISTICS =====")

print("Gyro Yaw mean:", np.mean(gy_deg))
print("Gyro Pitch mean:", np.mean(gp_deg))
print("Gyro Roll mean:", np.mean(gr_deg))

print("Gyro Yaw std:", np.std(gy_deg))
print("Gyro Pitch std:", np.std(gp_deg))
print("Gyro Roll std:", np.std(gr_deg))

print("\n===== CORRELATION WITH GPS TURN RATE =====")

print("Gyro Yaw:", corr_yaw)
print("Gyro Pitch:", corr_pitch)
print("Gyro Roll:", corr_roll)

# ------------------------------------------------------------
# Absolute correlation ranking
# ------------------------------------------------------------

correlations = {
    "Yaw": abs(corr_yaw),
    "Pitch": abs(corr_pitch),
    "Roll": abs(corr_roll)
}

ranking = sorted(
    correlations.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\n===== AXIS RANKING =====")

for axis, value in ranking:
    print(
        f"{axis:6s} -> |correlation| = {value:.6f}"
    )

best_axis = ranking[0][0]

print("\n===== RESULT =====")

print("Best gyro axis:", best_axis)
print(
    "Absolute correlation:",
    ranking[0][1]
)

# ------------------------------------------------------------
# Sign test for best axis
# ------------------------------------------------------------

gyro_values = {
    "Yaw": gy_deg,
    "Pitch": gp_deg,
    "Roll": gr_deg
}

best_values = gyro_values[best_axis]

same_mae = np.mean(
    np.abs(gps_rate - best_values)
)

opposite_mae = np.mean(
    np.abs(gps_rate + best_values)
)

print("\n===== SIGN TEST =====")

print("Same-direction MAE:", same_mae)
print("Opposite-direction MAE:", opposite_mae)

# ------------------------------------------------------------
# Regression scale
# GPS rate ≈ scale × gyro
# ------------------------------------------------------------

denominator = np.sum(best_values ** 2)

if denominator > 0:

    scale = np.sum(
        best_values * gps_rate
    ) / denominator

    predicted = scale * best_values

    regression_mae = np.mean(
        np.abs(gps_rate - predicted)
    )

else:

    scale = np.nan
    regression_mae = np.nan

print("\n===== SCALE RELATIONSHIP =====")

print("Estimated scale:", scale)
print("Regression MAE:", regression_mae)

print("\n========================================")
print("STEP 4E COMPLETE")
print("========================================")