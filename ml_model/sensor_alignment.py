import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PRIME
# Step 3: Phone-to-Vehicle Sensor Alignment
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n===== SENSOR ALIGNMENT ANALYSIS =====")
print("Total samples:", len(df))

# ------------------------------------------------------------
# 2. SELECT MOVING SAMPLES
# ------------------------------------------------------------

speed = df["GPS SPEED (Kmh)"]

moving = df[speed > 3].copy()

print("Moving samples:", len(moving))

# ------------------------------------------------------------
# 3. EXTRACT SENSOR DATA
# ------------------------------------------------------------

ax = moving["ACCELEROMETER X (m/s²)"].to_numpy()
ay = moving["ACCELEROMETER Y (m/s²)"].to_numpy()
az = moving["ACCELEROMETER Z (m/s²)"].to_numpy()

gx = moving["GYROSCOPE Yaw (rad/s)"].to_numpy()
gy = moving["GYROSCOPE Pitch (rad/s)"].to_numpy()
gz = moving["GYROSCOPE Roll (rad/s)"].to_numpy()

gravity_x = moving["GRAVITY X (m/s²)"].to_numpy()
gravity_y = moving["GRAVITY Y (m/s²)"].to_numpy()
gravity_z = moving["GRAVITY Z (m/s²)"].to_numpy()

mag_x = moving["MAGNETIC FIELD X (Î¼T)"].to_numpy()
mag_y = moving["MAGNETIC FIELD Y (Î¼T)"].to_numpy()
mag_z = moving["MAGNETIC FIELD Z (Î¼T)"].to_numpy()

gps_heading = moving["GPS ORIENTATION (Â°)"].to_numpy()
phone_yaw = moving["ORIENTATION (Yaw) (Â°)"].to_numpy()

# ------------------------------------------------------------
# 4. BASIC SENSOR STATISTICS
# ------------------------------------------------------------

print("\n===== ACCELEROMETER =====")

print("X mean:", np.mean(ax))
print("Y mean:", np.mean(ay))
print("Z mean:", np.mean(az))

print("\n===== GYROSCOPE =====")

print("Yaw mean:", np.mean(gx))
print("Pitch mean:", np.mean(gy))
print("Roll mean:", np.mean(gz))

print("\n===== GRAVITY =====")

print("Gravity X mean:", np.mean(gravity_x))
print("Gravity Y mean:", np.mean(gravity_y))
print("Gravity Z mean:", np.mean(gravity_z))

print("\n===== MAGNETOMETER =====")

print("Magnetic X mean:", np.mean(mag_x))
print("Magnetic Y mean:", np.mean(mag_y))
print("Magnetic Z mean:", np.mean(mag_z))

# ------------------------------------------------------------
# 5. SENSOR MAGNITUDES
# ------------------------------------------------------------

acc_magnitude = np.sqrt(
    ax**2 +
    ay**2 +
    az**2
)

mag_magnitude = np.sqrt(
    mag_x**2 +
    mag_y**2 +
    mag_z**2
)

gravity_magnitude = np.sqrt(
    gravity_x**2 +
    gravity_y**2 +
    gravity_z**2
)

print("\n===== SENSOR MAGNITUDES =====")

print("Acceleration magnitude mean:",
      np.mean(acc_magnitude))

print("Gravity magnitude mean:",
      np.mean(gravity_magnitude))

print("Magnetic field magnitude mean:",
      np.mean(mag_magnitude))

# ------------------------------------------------------------
# 6. HORIZONTAL MAGNETIC HEADING
# ------------------------------------------------------------

mag_heading = np.degrees(
    np.arctan2(mag_y, mag_x)
)

mag_heading = (mag_heading + 360) % 360

# ------------------------------------------------------------
# 7. ANGLE DIFFERENCE FUNCTION
# ------------------------------------------------------------

def angle_difference(a, b):
    return (a - b + 180) % 360 - 180

# ------------------------------------------------------------
# 8. COMPARE MAGNETIC HEADING WITH GPS HEADING
# ------------------------------------------------------------

mag_gps_difference = angle_difference(
    gps_heading,
    mag_heading
)

print("\n===== MAGNETOMETER vs GPS =====")

print("Mean difference:",
      np.mean(mag_gps_difference))

print("Median difference:",
      np.median(mag_gps_difference))

print("Mean absolute difference:",
      np.mean(np.abs(mag_gps_difference)))

# ------------------------------------------------------------
# 9. COMPARE PHONE YAW WITH GPS AGAIN
# ------------------------------------------------------------

phone_gps_difference = angle_difference(
    gps_heading,
    phone_yaw
)

print("\n===== PHONE YAW vs GPS =====")

print("Mean difference:",
      np.mean(phone_gps_difference))

print("Median difference:",
      np.median(phone_gps_difference))

print("Mean absolute difference:",
      np.mean(np.abs(phone_gps_difference)))

# ------------------------------------------------------------
# 10. PLOT GPS HEADING AND PHONE YAW
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    gps_heading,
    label="GPS Heading"
)

plt.plot(
    phone_yaw,
    label="Phone Yaw"
)

plt.xlabel("Sample")
plt.ylabel("Heading (degrees)")
plt.title("GPS Heading vs Phone Yaw")
plt.legend()
plt.grid()

plt.show()

# ------------------------------------------------------------
# 11. PLOT GPS HEADING AND MAGNETIC HEADING
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    gps_heading,
    label="GPS Heading"
)

plt.plot(
    mag_heading,
    label="Magnetic Heading"
)

plt.xlabel("Sample")
plt.ylabel("Heading (degrees)")
plt.title("GPS Heading vs Magnetic Heading")
plt.legend()
plt.grid()

plt.show()

# ------------------------------------------------------------
# 12. PLOT GRAVITY
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    gravity_x,
    label="Gravity X"
)

plt.plot(
    gravity_y,
    label="Gravity Y"
)

plt.plot(
    gravity_z,
    label="Gravity Z"
)

plt.xlabel("Sample")
plt.ylabel("Gravity (m/s²)")
plt.title("Gravity Components During Vehicle Motion")
plt.legend()
plt.grid()

plt.show()

# ------------------------------------------------------------
# 13. FINAL MESSAGE
# ------------------------------------------------------------

print("\n========================================")
print("STEP 3 ALIGNMENT DIAGNOSTICS COMPLETE")
print("========================================")

print("\nNext objective:")
print("Determine the correct phone coordinate frame")
print("and estimate the phone-to-vehicle transformation.")