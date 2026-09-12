import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 6 — IMU-ONLY DEAD RECKONING
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n========================================")
print("PROJECT PRIME — IMU-ONLY DEAD RECKONING")
print("========================================")

# ------------------------------------------------------------
# TIME
# ------------------------------------------------------------

time = df["TIME SINCE START (ms)"].to_numpy() / 1000.0

dt = np.diff(time)

valid_dt = (
    np.isfinite(dt)
    & (dt > 0)
    & (dt < 1.0)
)

# ------------------------------------------------------------
# ACCELEROMETER
# ------------------------------------------------------------

ax = df["ACCELEROMETER X (m/s²)"].to_numpy()
ay = df["ACCELEROMETER Y (m/s²)"].to_numpy()
az = df["ACCELEROMETER Z (m/s²)"].to_numpy()

# ------------------------------------------------------------
# GRAVITY
# ------------------------------------------------------------

gx = df["GRAVITY X (m/s²)"].to_numpy()
gy = df["GRAVITY Y (m/s²)"].to_numpy()
gz = df["GRAVITY Z (m/s²)"].to_numpy()

# ------------------------------------------------------------
# REMOVE GRAVITY
# ------------------------------------------------------------

lin_x = ax - gx
lin_y = ay - gy
lin_z = az - gz

# ------------------------------------------------------------
# INTEGRATE ACCELERATION
# ------------------------------------------------------------

vx = np.zeros(len(df))
vy = np.zeros(len(df))
vz = np.zeros(len(df))

px = np.zeros(len(df))
py = np.zeros(len(df))
pz = np.zeros(len(df))

for i in range(1, len(df)):

    dti = time[i] - time[i - 1]

    if not np.isfinite(dti) or dti <= 0 or dti >= 1.0:
        vx[i] = vx[i - 1]
        vy[i] = vy[i - 1]
        vz[i] = vz[i - 1]

        px[i] = px[i - 1]
        py[i] = py[i - 1]
        pz[i] = pz[i - 1]

        continue

    # velocity integration
    vx[i] = vx[i - 1] + lin_x[i] * dti
    vy[i] = vy[i - 1] + lin_y[i] * dti
    vz[i] = vz[i - 1] + lin_z[i] * dti

    # position integration
    px[i] = px[i - 1] + vx[i - 1] * dti + 0.5 * lin_x[i] * dti**2
    py[i] = py[i - 1] + vy[i - 1] * dti + 0.5 * lin_y[i] * dti**2
    pz[i] = pz[i - 1] + vz[i - 1] * dti + 0.5 * lin_z[i] * dti**2

# ------------------------------------------------------------
# IMU DISTANCE
# ------------------------------------------------------------

imu_distance = np.sqrt(
    px**2 +
    py**2 +
    pz**2
)

# ------------------------------------------------------------
# VELOCITY MAGNITUDE
# ------------------------------------------------------------

velocity_magnitude = np.sqrt(
    vx**2 +
    vy**2 +
    vz**2
)

# ------------------------------------------------------------
# GPS REFERENCE POSITION
# ------------------------------------------------------------

lat = df["GPS LATITUDE (degrees)"].to_numpy()
lon = df["GPS LONGITUDE (degrees)"].to_numpy()

earth_radius = 6371000.0

lat0 = lat[0]
lon0 = lon[0]

gps_north = (
    np.deg2rad(lat - lat0)
    * earth_radius
)

gps_east = (
    np.deg2rad(lon - lon0)
    * earth_radius
    * np.cos(np.deg2rad(lat0))
)

gps_distance = np.sqrt(
    gps_north**2 +
    gps_east**2
)

# ------------------------------------------------------------
# FINAL VALUES
# ------------------------------------------------------------

print("\n===== DATA =====")

print("Samples:", len(df))

print(
    "Duration:",
    time[-1] - time[0],
    "seconds"
)

print(
    "Duration:",
    (time[-1] - time[0]) / 60,
    "minutes"
)

print("\n===== IMU INTEGRATION =====")

print(
    "Final X position:",
    px[-1],
    "m"
)

print(
    "Final Y position:",
    py[-1],
    "m"
)

print(
    "Final Z position:",
    pz[-1],
    "m"
)

print(
    "Final IMU displacement:",
    imu_distance[-1],
    "m"
)

print(
    "Final IMU velocity:",
    velocity_magnitude[-1],
    "m/s"
)

print("\n===== GPS REFERENCE =====")

print(
    "Final GPS displacement:",
    gps_distance[-1],
    "m"
)

print("\n===== DRIFT =====")

print(
    "Position magnitude difference:",
    abs(imu_distance[-1] - gps_distance[-1]),
    "m"
)

print(
    "IMU/GPS displacement ratio:",
    imu_distance[-1] / max(gps_distance[-1], 1)
)

print("\n===== SENSOR BIAS CHECK =====")

print(
    "Linear acceleration X mean:",
    np.mean(lin_x)
)

print(
    "Linear acceleration Y mean:",
    np.mean(lin_y)
)

print(
    "Linear acceleration Z mean:",
    np.mean(lin_z)
)

print("\n========================================")
print("STEP 6 COMPLETE")
print("========================================")