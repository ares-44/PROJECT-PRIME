import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# --------------------------------------------------
# Basic columns
# --------------------------------------------------

speed = df["GPS SPEED (Kmh)"]

gps_heading = df["GPS ORIENTATION (Â°)"]

phone_yaw = df["ORIENTATION (Yaw) (Â°)"]

# --------------------------------------------------
# Keep only moving vehicle samples
# GPS heading is unreliable when nearly stationary.
# --------------------------------------------------

moving = df[speed > 3].copy()

print("\n===== ALIGNMENT ESTIMATION =====")

print("Total samples:", len(df))
print("Moving samples:", len(moving))

# --------------------------------------------------
# Calculate circular angle difference
# Result is between -180 and +180 degrees.
# --------------------------------------------------

def angle_difference(a, b):
    return (a - b + 180) % 360 - 180


heading_difference = angle_difference(
    moving["GPS ORIENTATION (Â°)"].to_numpy(),
    moving["ORIENTATION (Yaw) (Â°)"].to_numpy()
)

# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("\n===== HEADING DIFFERENCE =====")

print(
    "Mean:",
    np.mean(heading_difference)
)

print(
    "Median:",
    np.median(heading_difference)
)

print(
    "Standard deviation:",
    np.std(heading_difference)
)

print(
    "Minimum:",
    np.min(heading_difference)
)

print(
    "Maximum:",
    np.max(heading_difference)
)

# --------------------------------------------------
# Estimate a robust alignment offset
# Median is preferred because heading wraps around
# 0/360 degrees.
# --------------------------------------------------

alignment_offset = np.median(heading_difference)

print("\n===== ESTIMATED ALIGNMENT =====")

print(
    "Phone → vehicle heading offset:",
    alignment_offset,
    "degrees"
)

# --------------------------------------------------
# Apply estimated alignment
# --------------------------------------------------

moving["estimated_vehicle_heading"] = (
    moving["ORIENTATION (Yaw) (Â°)"]
    + alignment_offset
) % 360

# --------------------------------------------------
# Compare estimated heading with GPS heading
# --------------------------------------------------

moving["heading_error"] = angle_difference(
    moving["estimated_vehicle_heading"].to_numpy(),
    moving["GPS ORIENTATION (Â°)"].to_numpy()
)

print("\n===== AFTER ALIGNMENT =====")

print(
    "Mean absolute heading error:",
    np.mean(np.abs(moving["heading_error"]))
)

print(
    "Median absolute heading error:",
    np.median(np.abs(moving["heading_error"]))
)

print(
    "90th percentile absolute error:",
    np.percentile(
        np.abs(moving["heading_error"]),
        90
    )
)

# --------------------------------------------------
# Save alignment result
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\alignment_result.csv"

moving.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== SAVED =====")
print("Alignment data saved to:")
print(OUTPUT_PATH)