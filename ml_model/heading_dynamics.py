import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 4D — DYNAMIC HEADING RELATIONSHIP
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

speed = df["GPS SPEED (Kmh)"].to_numpy()

gps = df["GPS ORIENTATION (Â°)"].to_numpy()
yaw = df["ORIENTATION (Yaw) (Â°)"].to_numpy()

time = df["TIME SINCE START (ms)"].to_numpy()

# ------------------------------------------------------------
# Moving samples
# ------------------------------------------------------------

valid = (
    (speed > 5)
    & np.isfinite(gps)
    & np.isfinite(yaw)
)

gps = gps[valid]
yaw = yaw[valid]
time = time[valid]

# ------------------------------------------------------------
# Circular unwrap
# ------------------------------------------------------------

gps_rad = np.unwrap(np.deg2rad(gps))
yaw_rad = np.unwrap(np.deg2rad(yaw))

gps_unwrapped = np.rad2deg(gps_rad)
yaw_unwrapped = np.rad2deg(yaw_rad)

# ------------------------------------------------------------
# Heading change
# ------------------------------------------------------------

gps_change = np.diff(gps_unwrapped)
yaw_change = np.diff(yaw_unwrapped)

# ------------------------------------------------------------
# Remove extreme jumps/noise
# ------------------------------------------------------------

valid_change = (
    np.isfinite(gps_change)
    & np.isfinite(yaw_change)
    & (np.abs(gps_change) < 30)
    & (np.abs(yaw_change) < 30)
)

gps_change = gps_change[valid_change]
yaw_change = yaw_change[valid_change]

# ------------------------------------------------------------
# Correlation
# ------------------------------------------------------------

correlation = np.corrcoef(
    gps_change,
    yaw_change
)[0, 1]

# ------------------------------------------------------------
# Sign test
# ------------------------------------------------------------

positive_mae = np.mean(
    np.abs(gps_change - yaw_change)
)

negative_mae = np.mean(
    np.abs(gps_change + yaw_change)
)

# ------------------------------------------------------------
# Scale relationship
# ------------------------------------------------------------

slope = np.sum(
    yaw_change * gps_change
) / np.sum(
    yaw_change ** 2
)

predicted = slope * yaw_change

scaled_mae = np.mean(
    np.abs(gps_change - predicted)
)

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\n========================================")
print("PROJECT PRIME — HEADING DYNAMICS")
print("========================================")

print("Valid heading-change samples:", len(gps_change))

print("\n===== HEADING CHANGE STATISTICS =====")

print("GPS change mean:",
      np.mean(gps_change))

print("Phone yaw change mean:",
      np.mean(yaw_change))

print("GPS change std:",
      np.std(gps_change))

print("Phone yaw change std:",
      np.std(yaw_change))

print("\n===== DYNAMIC CORRELATION =====")

print("Correlation:", correlation)

print("\n===== SIGN TEST =====")

print(
    "Same-direction MAE:",
    positive_mae
)

print(
    "Opposite-direction MAE:",
    negative_mae
)

print("\n===== SCALE RELATIONSHIP =====")

print("Estimated scale:", slope)

print(
    "Scaled MAE:",
    scaled_mae
)

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

print("\n===== INTERPRETATION =====")

if correlation > 0.7:

    print(
        "STRONG: Phone yaw changes track GPS heading changes."
    )

elif correlation > 0.4:

    print(
        "MODERATE: Phone yaw contains useful dynamic information."
    )

else:

    print(
        "WEAK: Phone yaw does not reliably track vehicle heading."
    )

if positive_mae < negative_mae:

    print(
        "Direction: SAME"
    )

else:

    print(
        "Direction: OPPOSITE"
    )

print("\n========================================")
print("STEP 4D COMPLETE")
print("========================================")