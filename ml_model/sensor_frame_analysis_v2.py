import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 4B — PHONE → VEHICLE FRAME ANALYSIS
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

lat = df["GPS LATITUDE (degrees)"].to_numpy()
lon = df["GPS LONGITUDE (degrees)"].to_numpy()
speed = df["GPS SPEED (Kmh)"].to_numpy()

ax = df["ACCELEROMETER X (m/s²)"].to_numpy()
ay = df["ACCELEROMETER Y (m/s²)"].to_numpy()
az = df["ACCELEROMETER Z (m/s²)"].to_numpy()

gx = df["GRAVITY X (m/s²)"].to_numpy()
gy = df["GRAVITY Y (m/s²)"].to_numpy()
gz = df["GRAVITY Z (m/s²)"].to_numpy()

time = df["TIME SINCE START (ms)"].to_numpy() / 1000.0

# ------------------------------------------------------------
# Remove gravity
# ------------------------------------------------------------

linear_x = ax - gx
linear_y = ay - gy
linear_z = az - gz

# ------------------------------------------------------------
# Convert GPS coordinates to local displacement
# Approximation is valid for a local trajectory.
# ------------------------------------------------------------

lat0 = np.nanmedian(lat)
lon0 = np.nanmedian(lon)

earth_radius = 6371000.0

north = np.deg2rad(lat - lat0) * earth_radius
east = (
    np.deg2rad(lon - lon0)
    * earth_radius
    * np.cos(np.deg2rad(lat0))
)

# ------------------------------------------------------------
# GPS movement direction
# ------------------------------------------------------------

dn = np.gradient(north)
de = np.gradient(east)

gps_motion_heading = np.degrees(
    np.arctan2(de, dn)
)

gps_motion_heading = (
    gps_motion_heading + 360
) % 360

# ------------------------------------------------------------
# Keep moving samples
# ------------------------------------------------------------

moving = (
    (speed > 5)
    & np.isfinite(north)
    & np.isfinite(east)
    & np.isfinite(linear_x)
    & np.isfinite(linear_y)
    & np.isfinite(linear_z)
)

print("\n========================================")
print("PROJECT PRIME — SENSOR FRAME V2")
print("========================================")

print("Total samples:", len(df))
print("Moving samples:", np.sum(moving))

# ------------------------------------------------------------
# Statistics of gravity-removed acceleration
# ------------------------------------------------------------

lx = linear_x[moving]
ly = linear_y[moving]
lz = linear_z[moving]

print("\n===== LINEAR ACCELERATION =====")

print("X mean:", np.mean(lx))
print("Y mean:", np.mean(ly))
print("Z mean:", np.mean(lz))

print("X std:", np.std(lx))
print("Y std:", np.std(ly))
print("Z std:", np.std(lz))

# ------------------------------------------------------------
# Horizontal acceleration magnitude
# ------------------------------------------------------------

horizontal_mag = np.sqrt(
    lx**2 + ly**2
)

print("\n===== HORIZONTAL ACCELERATION =====")

print("Mean:", np.mean(horizontal_mag))
print("Median:", np.median(horizontal_mag))
print("90th percentile:",
      np.percentile(horizontal_mag, 90))

# ------------------------------------------------------------
# Determine dominant horizontal axis
# ------------------------------------------------------------

x_std = np.std(lx)
y_std = np.std(ly)

print("\n===== HORIZONTAL AXIS TEST =====")

print("X variation:", x_std)
print("Y variation:", y_std)

if x_std > y_std:
    dominant_axis = "X"
else:
    dominant_axis = "Y"

print("Dominant horizontal axis:", dominant_axis)

# ------------------------------------------------------------
# Test relationship between GPS motion direction
# and phone acceleration direction.
#
# We don't assume the phone yaw is correct.
# ------------------------------------------------------------

phone_acc_heading = np.degrees(
    np.arctan2(ly, lx)
)

phone_acc_heading = (
    phone_acc_heading + 360
) % 360

gps_heading = gps_motion_heading[moving]

# Circular difference
difference = (
    phone_acc_heading
    - gps_heading
    + 180
) % 360 - 180

print("\n===== ACCELERATION DIRECTION vs GPS MOTION =====")

print("Mean difference:",
      np.mean(difference))

print("Median difference:",
      np.median(difference))

print("Mean absolute difference:",
      np.mean(np.abs(difference)))

print("Median absolute difference:",
      np.median(np.abs(difference)))

# ------------------------------------------------------------
# Test simple axis transformations.
#
# [X,Y]
# [-X,Y]
# [X,-Y]
# [-X,-Y]
# [Y,X]
# [-Y,X]
# [Y,-X]
# [-Y,-X]
# ------------------------------------------------------------

transformations = {
    "X,Y": (lx, ly),
    "-X,Y": (-lx, ly),
    "X,-Y": (lx, -ly),
    "-X,-Y": (-lx, -ly),
    "Y,X": (ly, lx),
    "-Y,X": (-ly, lx),
    "Y,-X": (ly, -lx),
    "-Y,-X": (-ly, -lx),
}

results = []

for name, (tx, ty) in transformations.items():

    heading = np.degrees(
        np.arctan2(ty, tx)
    )

    heading = (
        heading + 360
    ) % 360

    diff = (
        heading
        - gps_heading
        + 180
    ) % 360 - 180

    mae = np.mean(np.abs(diff))

    results.append(
        (name, mae)
    )

results.sort(key=lambda x: x[1])

print("\n===== AXIS TRANSFORMATION TEST =====")

for name, mae in results:
    print(
        f"{name:8s} -> MAE: {mae:.3f} degrees"
    )

best_name, best_mae = results[0]

print("\n===== BEST CANDIDATE =====")

print("Transformation:", best_name)
print("MAE:", best_mae)

print("\n========================================")
print("STEP 4B COMPLETE")
print("========================================")