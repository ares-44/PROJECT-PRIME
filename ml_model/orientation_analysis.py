import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 4C — ORIENTATION CONVENTION ANALYSIS
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

speed = df["GPS SPEED (Kmh)"].to_numpy()

gps = df["GPS ORIENTATION (Â°)"].to_numpy()
yaw = df["ORIENTATION (Yaw) (Â°)"].to_numpy()

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

# ------------------------------------------------------------
# Circular difference
# ------------------------------------------------------------

def angle_diff(a, b):
    return (a - b + 180) % 360 - 180


def circular_mae(a, b):
    diff = angle_diff(a, b)
    return np.mean(np.abs(diff))


# ------------------------------------------------------------
# Test different yaw conventions
# ------------------------------------------------------------

tests = {}

# Basic
tests["yaw"] = yaw
tests["-yaw"] = (-yaw) % 360

# 90 degree rotations
tests["yaw + 90"] = (yaw + 90) % 360
tests["yaw - 90"] = (yaw - 90) % 360

tests["-yaw + 90"] = (-yaw + 90) % 360
tests["-yaw - 90"] = (-yaw - 90) % 360

# 180 degree reversal
tests["yaw + 180"] = (yaw + 180) % 360
tests["-yaw + 180"] = (-yaw + 180) % 360

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

results = []

for name, candidate in tests.items():

    mae = circular_mae(candidate, gps)

    median_error = np.median(
        np.abs(angle_diff(candidate, gps))
    )

    results.append(
        (name, mae, median_error)
    )

results.sort(key=lambda x: x[1])

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\n========================================")
print("PROJECT PRIME — ORIENTATION ANALYSIS")
print("========================================")

print("Moving samples:", len(gps))

print("\n===== CONVENTION TEST =====")

for name, mae, median in results:

    print(
        f"{name:15s} "
        f"MAE: {mae:8.3f}°   "
        f"Median: {median:8.3f}°"
    )

# ------------------------------------------------------------
# Estimate best CONSTANT offset for every convention
# ------------------------------------------------------------

print("\n===== OPTIMAL OFFSET TEST =====")

offset_results = []

for name, candidate in tests.items():

    difference = angle_diff(
        gps,
        candidate
    )

    # Circular mean is more appropriate than ordinary mean
    radians = np.deg2rad(difference)

    circular_mean = np.arctan2(
        np.mean(np.sin(radians)),
        np.mean(np.cos(radians))
    )

    offset = np.rad2deg(circular_mean)

    aligned = (
        candidate + offset
    ) % 360

    error = angle_diff(
        aligned,
        gps
    )

    mae = np.mean(np.abs(error))
    median = np.median(np.abs(error))

    offset_results.append(
        (name, offset, mae, median)
    )

offset_results.sort(key=lambda x: x[2])

for name, offset, mae, median in offset_results:

    print(
        f"{name:15s} "
        f"Offset: {offset:8.3f}°   "
        f"MAE: {mae:8.3f}°   "
        f"Median: {median:8.3f}°"
    )

# ------------------------------------------------------------
# Final candidate
# ------------------------------------------------------------

best = offset_results[0]

print("\n===== BEST CONVENTION =====")

print("Convention:", best[0])
print("Estimated offset:", best[1], "degrees")
print("MAE after offset:", best[2], "degrees")
print("Median error:", best[3], "degrees")

print("\n========================================")
print("STEP 4C COMPLETE")
print("========================================")