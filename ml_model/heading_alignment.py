import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\gps_heading_v2_result.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# --------------------------------------------------
# 1. COLUMNS
# --------------------------------------------------

gps_heading = df["GPS_TRAJECTORY_HEADING_V2"]
phone_yaw = df["ORIENTATION (Yaw) (Â°)"]
speed = df["GPS SPEED (Kmh)"]

# --------------------------------------------------
# 2. RELIABLE MOVING SAMPLES
# --------------------------------------------------

valid = df[
    (speed > 5)
    & gps_heading.notna()
    & phone_yaw.notna()
].copy()

print("\n========================================")
print("PHONE → VEHICLE HEADING ALIGNMENT")
print("========================================")

print("\nTotal samples:", len(df))
print("Reliable moving samples:", len(valid))

# --------------------------------------------------
# 3. ANGLE DIFFERENCE
# --------------------------------------------------

def angle_difference(a, b):
    return (a - b + 180) % 360 - 180


gps = valid["GPS_TRAJECTORY_HEADING_V2"].to_numpy()
yaw = valid["ORIENTATION (Yaw) (Â°)"].to_numpy()

# GPS heading - phone yaw
difference = angle_difference(gps, yaw)

valid["heading_difference"] = difference

# --------------------------------------------------
# 4. BASIC STATISTICS
# --------------------------------------------------

print("\n===== RAW HEADING DIFFERENCE =====")

print(
    "Mean:",
    np.mean(difference)
)

print(
    "Median:",
    np.median(difference)
)

print(
    "Standard deviation:",
    np.std(difference)
)

print(
    "Minimum:",
    np.min(difference)
)

print(
    "Maximum:",
    np.max(difference)
)

# --------------------------------------------------
# 5. CIRCULAR MEAN OFFSET
# --------------------------------------------------

difference_rad = np.radians(difference)

circular_offset = np.degrees(
    np.arctan2(
        np.mean(np.sin(difference_rad)),
        np.mean(np.cos(difference_rad))
    )
)

circular_offset = (
    circular_offset + 360
) % 360

# Convert to signed range [-180, 180]
if circular_offset > 180:
    circular_offset -= 360

print("\n===== CIRCULAR ALIGNMENT =====")

print(
    "Estimated phone → vehicle heading offset:",
    circular_offset,
    "degrees"
)

# --------------------------------------------------
# 6. APPLY ALIGNMENT
# --------------------------------------------------

valid["aligned_vehicle_heading"] = (
    valid["ORIENTATION (Yaw) (Â°)"]
    + circular_offset
) % 360

# --------------------------------------------------
# 7. ERROR BEFORE ALIGNMENT
# --------------------------------------------------

error_before = angle_difference(
    valid["ORIENTATION (Yaw) (Â°)"].to_numpy(),
    valid["GPS_TRAJECTORY_HEADING_V2"].to_numpy()
)

# --------------------------------------------------
# 8. ERROR AFTER ALIGNMENT
# --------------------------------------------------

error_after = angle_difference(
    valid["aligned_vehicle_heading"].to_numpy(),
    valid["GPS_TRAJECTORY_HEADING_V2"].to_numpy()
)

print("\n===== BEFORE ALIGNMENT =====")

print(
    "Mean absolute error:",
    np.mean(np.abs(error_before))
)

print(
    "Median absolute error:",
    np.median(np.abs(error_before))
)

print(
    "90th percentile error:",
    np.percentile(
        np.abs(error_before),
        90
    )
)

print("\n===== AFTER ALIGNMENT =====")

print(
    "Mean absolute error:",
    np.mean(np.abs(error_after))
)

print(
    "Median absolute error:",
    np.median(np.abs(error_after))
)

print(
    "90th percentile error:",
    np.percentile(
        np.abs(error_after),
        90
    )
)

# --------------------------------------------------
# 9. IMPROVEMENT
# --------------------------------------------------

before_mae = np.mean(np.abs(error_before))
after_mae = np.mean(np.abs(error_after))

improvement = (
    (before_mae - after_mae)
    / before_mae
) * 100

print("\n===== ALIGNMENT IMPROVEMENT =====")

print(
    "MAE improvement:",
    improvement,
    "%"
)

# --------------------------------------------------
# 10. PLOT BEFORE ALIGNMENT
# --------------------------------------------------

plot_samples = valid.iloc[:5000]

plt.figure(figsize=(12, 5))

plt.plot(
    plot_samples["time_seconds"],
    plot_samples["GPS_TRAJECTORY_HEADING_V2"],
    label="GNSS Vehicle Heading"
)

plt.plot(
    plot_samples["time_seconds"],
    plot_samples["ORIENTATION (Yaw) (Â°)"],
    label="Raw Phone Yaw"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")
plt.title("Before Phone → Vehicle Alignment")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 11. PLOT AFTER ALIGNMENT
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    plot_samples["time_seconds"],
    plot_samples["GPS_TRAJECTORY_HEADING_V2"],
    label="GNSS Vehicle Heading"
)

plt.plot(
    plot_samples["time_seconds"],
    plot_samples["aligned_vehicle_heading"],
    label="Aligned Phone Heading"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading (degrees)")
plt.title("After Phone → Vehicle Alignment")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 12. HEADING ERROR
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    plot_samples["time_seconds"],
    error_before[:len(plot_samples)],
    label="Error Before Alignment"
)

plt.plot(
    plot_samples["time_seconds"],
    error_after[:len(plot_samples)],
    label="Error After Alignment"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Heading Error (degrees)")
plt.title("Heading Alignment Error")

plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 13. SAVE
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\heading_alignment_result.csv"

valid.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("HEADING ALIGNMENT COMPLETE")
print("========================================")

print("\nSaved result:")
print(OUTPUT_PATH)