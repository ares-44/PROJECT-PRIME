# ============================================================
# PROJECT PRIME
# PROPER PHONE -> VEHICLE ALIGNMENT
#
# Method:
#   GNSS speed + GNSS trajectory
#              +
#   Smartphone accelerometer + gravity
#
# Goal:
#   Estimate the smartphone horizontal forward axis
#   relative to the vehicle's direction of travel.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME"
    r"\data\cleaned_S-S1.csv"
)

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("\n===== DATA LOADED =====")
print("Total samples:", len(df))


# ============================================================
# 2. TIME
# ============================================================

df["time_seconds"] = (
    df["TIME SINCE START (ms)"] / 1000.0
)

time = df["time_seconds"].to_numpy()


# ============================================================
# 3. EXTRACT GPS DATA
# ============================================================

gps_speed_kmh = df[
    "GPS SPEED (Kmh)"
].to_numpy()

gps_lat = df[
    "GPS LATITUDE (degrees)"
].to_numpy()

gps_lon = df[
    "GPS LONGITUDE (degrees)"
].to_numpy()

gps_orientation = df[
    "GPS ORIENTATION (Â°)"
].to_numpy()


# ============================================================
# 4. EXTRACT ACCELEROMETER
# ============================================================

acc = df[
    [
        "ACCELEROMETER X (m/s²)",
        "ACCELEROMETER Y (m/s²)",
        "ACCELEROMETER Z (m/s²)"
    ]
].to_numpy()


# ============================================================
# 5. EXTRACT GRAVITY
# ============================================================

gravity = df[
    [
        "GRAVITY X (m/s²)",
        "GRAVITY Y (m/s²)",
        "GRAVITY Z (m/s²)"
    ]
].to_numpy()


# ============================================================
# 6. REMOVE GRAVITY
#
# Linear acceleration = measured acceleration - gravity
# ============================================================

linear_acc = acc - gravity


print("\n===== LINEAR ACCELERATION =====")

print(
    "X mean:",
    np.mean(linear_acc[:, 0])
)

print(
    "Y mean:",
    np.mean(linear_acc[:, 1])
)

print(
    "Z mean:",
    np.mean(linear_acc[:, 2])
)


# ============================================================
# 7. CONVERT GPS SPEED TO m/s
# ============================================================

gps_speed_ms = (
    gps_speed_kmh / 3.6
)


# ============================================================
# 8. GPS SPEED INTERPOLATION
#
# GPS updates at approximately 1 Hz.
# IMU is approximately 10 Hz.
#
# Interpolate GPS speed onto the full IMU timeline.
# ============================================================

valid_gps_speed = np.isfinite(
    gps_speed_ms
)

gps_time = time[
    valid_gps_speed
]

gps_speed_valid = gps_speed_ms[
    valid_gps_speed
]


gps_speed_interp = np.interp(
    time,
    gps_time,
    gps_speed_valid
)


# ============================================================
# 9. SMOOTH GPS SPEED
#
# GPS speed can contain small jumps.
# Use rolling average.
# ============================================================

speed_series = pd.Series(
    gps_speed_interp
)

smooth_speed = (
    speed_series
    .rolling(
        window=11,
        center=True,
        min_periods=1
    )
    .mean()
    .to_numpy()
)


# ============================================================
# 10. ESTIMATE VEHICLE LONGITUDINAL ACCELERATION
# ============================================================

dt = np.gradient(time)

dt[
    dt <= 0
] = np.nan


gps_longitudinal_acc = (
    np.gradient(
        smooth_speed,
        time
    )
)


# Smooth again
gps_acc_series = pd.Series(
    gps_longitudinal_acc
)

gps_longitudinal_acc = (
    gps_acc_series
    .rolling(
        window=11,
        center=True,
        min_periods=1
    )
    .mean()
    .to_numpy()
)


# ============================================================
# 11. CREATE GRAVITY UNIT VECTOR
# ============================================================

gravity_norm = np.linalg.norm(
    gravity,
    axis=1,
    keepdims=True
)

gravity_norm[
    gravity_norm < 1e-6
] = 1

gravity_unit = (
    gravity / gravity_norm
)


# ============================================================
# 12. REMOVE ANY REMAINING VERTICAL COMPONENT
#
# Project linear acceleration onto the horizontal plane.
# ============================================================

vertical_component = (
    np.sum(
        linear_acc * gravity_unit,
        axis=1,
        keepdims=True
    )
    * gravity_unit
)


horizontal_acc = (
    linear_acc - vertical_component
)


# ============================================================
# 13. HORIZONTAL ACCELERATION MAGNITUDE
# ============================================================

horizontal_acc_magnitude = np.linalg.norm(
    horizontal_acc,
    axis=1
)


# ============================================================
# 14. SELECT STRONG ACCELERATION EVENTS
#
# We need periods where vehicle acceleration is strong enough
# to reveal the longitudinal direction.
# ============================================================

strong_motion = (
    np.abs(gps_longitudinal_acc) > 0.20
)

moving = (
    gps_speed_kmh > 5
)

valid = (
    strong_motion
    & moving
    & np.isfinite(
        horizontal_acc_magnitude
    )
)


print("\n===== MOTION SELECTION =====")

print(
    "Moving samples:",
    np.sum(moving)
)

print(
    "Strong longitudinal acceleration samples:",
    np.sum(strong_motion)
)

print(
    "Final valid samples:",
    np.sum(valid)
)


# ============================================================
# 15. NORMALIZE HORIZONTAL ACCELERATION
# ============================================================

horizontal_unit = np.zeros_like(
    horizontal_acc
)

magnitude = (
    horizontal_acc_magnitude
)

safe = magnitude > 0.05

horizontal_unit[safe] = (
    horizontal_acc[safe]
    / magnitude[safe, None]
)


# ============================================================
# 16. DETERMINE ACCELERATION SIGN
#
# During acceleration:
#     acceleration points forward.
#
# During braking:
#     acceleration points backward.
#
# GPS longitudinal acceleration tells us which is which.
# ============================================================

acc_direction = (
    np.sign(
        gps_longitudinal_acc
    )
)


# Correct horizontal vector direction
# so that positive longitudinal acceleration
# corresponds to vehicle forward.

vehicle_directed_acc = (
    horizontal_unit
    * acc_direction[:, None]
)


# ============================================================
# 17. ESTIMATE PHONE FORWARD AXIS
#
# Calculate average directed acceleration vector.
# ============================================================

valid_vectors = (
    vehicle_directed_acc[valid]
)


mean_forward_vector = np.mean(
    valid_vectors,
    axis=0
)


mean_forward_norm = np.linalg.norm(
    mean_forward_vector
)


if mean_forward_norm > 0:

    mean_forward_unit = (
        mean_forward_vector
        / mean_forward_norm
    )

else:

    mean_forward_unit = (
        np.array(
            [1.0, 0.0, 0.0]
        )
    )


print("\n===== ESTIMATED PHONE FORWARD VECTOR =====")

print(
    "X:",
    mean_forward_unit[0]
)

print(
    "Y:",
    mean_forward_unit[1]
)

print(
    "Z:",
    mean_forward_unit[2]
)


# ============================================================
# 18. FIND DOMINANT PHONE AXIS
# ============================================================

axis_names = [
    "X",
    "Y",
    "Z"
]

dominant_axis_index = np.argmax(
    np.abs(
        mean_forward_unit
    )
)

dominant_axis = (
    axis_names[
        dominant_axis_index
    ]
)

dominant_sign = np.sign(
    mean_forward_unit[
        dominant_axis_index
    ]
)


print("\n===== DOMINANT PHONE AXIS =====")

print(
    "Axis:",
    dominant_axis
)

print(
    "Sign:",
    dominant_sign
)


# ============================================================
# 19. BUILD PHONE FORWARD HEADING
#
# We now calculate the direction of the horizontal
# acceleration vector inside the phone frame.
# ============================================================

phone_forward_x = (
    vehicle_directed_acc[:, 0]
)

phone_forward_y = (
    vehicle_directed_acc[:, 1]
)


phone_heading = np.degrees(
    np.arctan2(
        phone_forward_y,
        phone_forward_x
    )
)

phone_heading = (
    phone_heading + 360
) % 360


# ============================================================
# 20. CALCULATE GPS TRAJECTORY HEADING
#
# Use position changes over approximately 1 second.
# This avoids the previous problem where consecutive
# 10 Hz GPS rows were almost identical.
# ============================================================

lat = np.radians(
    gps_lat
)

lon = np.radians(
    gps_lon
)

earth_radius = 6371000.0


delta_lat = np.diff(lat)

delta_lon = np.diff(lon)


mean_lat = (
    lat[:-1] + lat[1:]
) / 2


north_distance = (
    delta_lat
    * earth_radius
)


east_distance = (
    delta_lon
    * earth_radius
    * np.cos(mean_lat)
)


distance = np.sqrt(
    north_distance ** 2
    + east_distance ** 2
)


trajectory_heading = np.zeros(
    len(df)
)

trajectory_heading[1:] = (
    np.degrees(
        np.arctan2(
            east_distance,
            north_distance
        )
    )
    + 360
) % 360


# ============================================================
# 21. SMOOTH GPS TRAJECTORY HEADING
# ============================================================

heading_rad = np.radians(
    trajectory_heading
)

heading_x = pd.Series(
    np.cos(heading_rad)
).rolling(
    window=11,
    center=True,
    min_periods=1
).mean()

heading_y = pd.Series(
    np.sin(heading_rad)
).rolling(
    window=11,
    center=True,
    min_periods=1
).mean()


trajectory_heading_smooth = (
    np.degrees(
        np.arctan2(
            heading_y,
            heading_x
        )
    )
    + 360
) % 360


# ============================================================
# 22. ALIGN PHONE ACCELERATION HEADING
#     WITH GPS VEHICLE HEADING
# ============================================================

alignment_valid = (
    valid
    & np.isfinite(
        phone_heading
    )
    & np.isfinite(
        trajectory_heading_smooth
    )
)


phone_h = phone_heading[
    alignment_valid
]

gps_h = trajectory_heading_smooth[
    alignment_valid
]


# ============================================================
# 23. ANGLE DIFFERENCE
# ============================================================

def angle_difference(a, b):

    return (
        (a - b + 180)
        % 360
    ) - 180


raw_difference = (
    angle_difference(
        gps_h,
        phone_h
    )
)


# ============================================================
# 24. ESTIMATE CIRCULAR OFFSET
# ============================================================

difference_rad = np.radians(
    raw_difference
)


alignment_offset = np.degrees(
    np.arctan2(
        np.mean(
            np.sin(
                difference_rad
            )
        ),
        np.mean(
            np.cos(
                difference_rad
            )
        )
    )
)


print("\n===== PHONE → VEHICLE ALIGNMENT =====")

print(
    "Estimated heading offset:",
    alignment_offset,
    "degrees"
)


# ============================================================
# 25. APPLY ALIGNMENT
# ============================================================

aligned_phone_heading = (
    phone_h
    + alignment_offset
) % 360


# ============================================================
# 26. CALCULATE ERROR
# ============================================================

heading_error = angle_difference(
    aligned_phone_heading,
    gps_h
)

absolute_error = np.abs(
    heading_error
)


mae = np.mean(
    absolute_error
)

median_error = np.median(
    absolute_error
)

p90_error = np.percentile(
    absolute_error,
    90
)


print("\n===== ALIGNMENT RESULTS =====")

print(
    "MAE:",
    mae,
    "degrees"
)

print(
    "Median error:",
    median_error,
    "degrees"
)

print(
    "90th percentile:",
    p90_error,
    "degrees"
)


# ============================================================
# 27. SAVE RESULTS
# ============================================================

result = df.loc[
    alignment_valid
].copy()


result[
    "GPS_TRAJECTORY_HEADING"
] = gps_h

result[
    "PHONE_ACCEL_HEADING"
] = phone_h

result[
    "ALIGNED_PHONE_HEADING"
] = aligned_phone_heading

result[
    "HEADING_ERROR"
] = heading_error

result[
    "ABS_HEADING_ERROR"
] = absolute_error


OUTPUT_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME"
    r"\data\proper_alignment_result.csv"
)


result.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n===== SAVED =====")

print(
    "Result saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 28. GRAPH — GPS VS PHONE HEADING
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    gps_h,
    label="GNSS Trajectory Heading"
)

plt.plot(
    aligned_phone_heading,
    label="Aligned Phone Heading",
    alpha=0.7
)

plt.xlabel(
    "Valid Sample"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "GNSS Heading vs Aligned Phone Heading"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 29. GRAPH — HEADING ERROR
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    heading_error
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Valid Sample"
)

plt.ylabel(
    "Heading Error (degrees)"
)

plt.title(
    "Phone → Vehicle Heading Error"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 30. GRAPH — GPS SPEED VS ACCELERATION
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    time,
    smooth_speed,
    label="GPS Speed (m/s)"
)

plt.plot(
    time,
    gps_longitudinal_acc,
    label="GPS Longitudinal Acceleration"
)

plt.xlabel(
    "Time (seconds)"
)

plt.title(
    "GNSS Speed and Estimated Longitudinal Acceleration"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print("\n")
print("====================================================")
print("      PROJECT PRIME ALIGNMENT COMPLETE")
print("====================================================")

print(
    "\nEstimated phone forward vector:"
)

print(
    mean_forward_unit
)

print(
    "\nDominant phone axis:",
    dominant_axis
)

print(
    "Axis sign:",
    dominant_sign
)

print(
    "\nPhone → Vehicle heading offset:",
    alignment_offset,
    "degrees"
)

print(
    "\nFinal MAE:",
    mae,
    "degrees"
)

print(
    "Final median error:",
    median_error,
    "degrees"
)

print(
    "Final 90th percentile:",
    p90_error,
    "degrees"
)

print(
    "\n===================================================="
)