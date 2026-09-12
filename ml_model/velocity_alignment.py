# ============================================================
# PROJECT PRIME
# GNSS VELOCITY BASED PHONE -> VEHICLE ALIGNMENT
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

time = (
    df["TIME SINCE START (ms)"].to_numpy()
    / 1000.0
)


# ============================================================
# 3. GPS DATA
# ============================================================

lat = np.radians(
    df["GPS LATITUDE (degrees)"].to_numpy()
)
lat = np.radians(
    df["GPS LATITUDE (degrees)"].to_numpy()
)

lon = np.radians(
    df["GPS LONGITUDE (degrees)"].to_numpy()
)


gps_speed = df[
    "GPS SPEED (Kmh)"
].to_numpy()

gps_orientation = df[
    "GPS ORIENTATION (Â°)"
].to_numpy()


# ============================================================
# 4. PHONE ACCELEROMETER
# ============================================================

acc = df[
    [
        "ACCELEROMETER X (m/s²)",
        "ACCELEROMETER Y (m/s²)",
        "ACCELEROMETER Z (m/s²)"
    ]
].to_numpy()


# ============================================================
# 5. GRAVITY
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
# ============================================================

linear_acc = (
    acc - gravity
)


# ============================================================
# 7. PROJECT ACCELERATION ONTO HORIZONTAL PLANE
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
# 8. SMOOTH PHONE ACCELERATION
# ============================================================

horizontal_acc_df = pd.DataFrame(
    horizontal_acc,
    columns=["X", "Y", "Z"]
)

horizontal_acc_smooth = (
    horizontal_acc_df
    .rolling(
        window=11,
        center=True,
        min_periods=1
    )
    .mean()
    .to_numpy()
)


# ============================================================
# 9. COMPUTE GNSS VELOCITY VECTOR
#
# Use a window instead of consecutive samples.
#
# Window = approximately 1 second.
# ============================================================

WINDOW = 10


north_velocity = np.full(
    len(df),
    np.nan
)

east_velocity = np.full(
    len(df),
    np.nan
)


earth_radius = 6371000.0


for i in range(WINDOW, len(df)):

    dt = (
        time[i]
        - time[i - WINDOW]
    )

    if dt <= 0:
        continue

    dlat = (
        lat[i]
        - lat[i - WINDOW]
    )

    dlon = (
        lon[i]
        - lon[i - WINDOW]
    )

    mean_lat = (
        lat[i]
        + lat[i - WINDOW]
    ) / 2

    north_distance = (
        dlat
        * earth_radius
    )

    east_distance = (
        dlon
        * earth_radius
        * np.cos(mean_lat)
    )

    north_velocity[i] = (
        north_distance / dt
    )

    east_velocity[i] = (
        east_distance / dt
    )


# ============================================================
# 10. GNSS TRAVEL HEADING
# ============================================================

gnss_heading = (
    np.degrees(
        np.arctan2(
            east_velocity,
            north_velocity
        )
    )
    + 360
) % 360


# ============================================================
# 11. SMOOTH GNSS HEADING
# ============================================================

heading_rad = np.radians(
    gnss_heading
)

heading_x = (
    pd.Series(
        np.cos(heading_rad)
    )
    .rolling(
        window=21,
        center=True,
        min_periods=1
    )
    .mean()
)

heading_y = (
    pd.Series(
        np.sin(heading_rad)
    )
    .rolling(
        window=21,
        center=True,
        min_periods=1
    )
    .mean()
)

gnss_heading_smooth = (
    np.degrees(
        np.arctan2(
            heading_y,
            heading_x
        )
    )
    + 360
) % 360


# ============================================================
# 12. COMPARE GNSS CALCULATED HEADING
#     WITH DATASET GPS ORIENTATION
# ============================================================

heading_difference = (
    (gnss_heading_smooth
     - gps_orientation
     + 180)
    % 360
) - 180


valid_heading = (
    (gps_speed > 5)
    & np.isfinite(
        gnss_heading_smooth
    )
    & np.isfinite(
        gps_orientation
    )
)

heading_error = np.abs(
    heading_difference[
        valid_heading
    ]
)


print("\n===== GNSS VELOCITY CHECK =====")

print(
    "Valid heading samples:",
    np.sum(valid_heading)
)

print(
    "GNSS heading vs dataset GPS orientation MAE:",
    np.mean(heading_error),
    "degrees"
)

print(
    "Median:",
    np.median(heading_error),
    "degrees"
)

print(
    "P90:",
    np.percentile(
        heading_error,
        90
    ),
    "degrees"
)


# ============================================================
# 13. PHONE HORIZONTAL ACCELERATION DIRECTION
#
# Normalize horizontal acceleration.
# ============================================================

acc_magnitude = np.linalg.norm(
    horizontal_acc_smooth,
    axis=1
)


acc_unit = np.zeros_like(
    horizontal_acc_smooth
)


safe = (
    acc_magnitude > 0.15
)

acc_unit[safe] = (
    horizontal_acc_smooth[safe]
    / acc_magnitude[
        safe,
        None
    ]
)


# ============================================================
# 14. SELECT STRONG MOTION
# ============================================================

valid_motion = (
    (gps_speed > 5)
    & (acc_magnitude > 0.15)
    & np.isfinite(
        gnss_heading_smooth
    )
)


print("\n===== PHONE MOTION =====")

print(
    "Valid motion samples:",
    np.sum(valid_motion)
)


# ============================================================
# 15. TEST EACH PHONE AXIS
#
# Instead of assuming X/Y/Z is forward,
# test every horizontal axis.
# ============================================================

axis_results = []


def angle_difference(a, b):

    return (
        (a - b + 180)
        % 360
    ) - 180


for axis_index, axis_name in enumerate(
    ["X", "Y", "Z"]
):

    # Use positive axis
    axis_signal = (
        acc_unit[
            valid_motion,
            axis_index
        ]
    )

    # Convert sign into direction
    axis_heading_component = (
        axis_signal
    )

    # Determine correlation with GNSS heading
    # through vector projection.

    gps_heading_rad = np.radians(
        gnss_heading_smooth[
            valid_motion
        ]
    )

    gps_north = np.cos(
        gps_heading_rad
    )

    gps_east = np.sin(
        gps_heading_rad
    )

    # Measure relationship using absolute correlation
    correlation = np.corrcoef(
        axis_heading_component,
        gps_north
    )[0, 1]

    axis_results.append({

        "axis": axis_name,

        "correlation": correlation,

        "mean_abs_signal": np.mean(
            np.abs(
                axis_heading_component
            )
        )
    })


axis_results_df = pd.DataFrame(
    axis_results
)


print(
    "\n===== PHONE AXIS TEST ====="
)

print(
    axis_results_df.to_string(
        index=False
    )
)


# ============================================================
# 16. USE VEHICLE HEADING TO CONSTRUCT
#     NORTH/EAST VELOCITY UNIT VECTOR
# ============================================================

gps_h = gnss_heading_smooth[
    valid_motion
]

gps_rad = np.radians(
    gps_h
)


vehicle_north = np.cos(
    gps_rad
)

vehicle_east = np.sin(
    gps_rad
)


# ============================================================
# 17. ESTIMATE PHONE ACCELERATION HEADING
#
# We cannot directly interpret phone X/Y as north/east.
# Instead, test the four horizontal orientations:
#
# X/Y
# -X/Y
# X/-Y
# -X/-Y
#
# ============================================================

phone_x = acc_unit[
    valid_motion,
    0
]

phone_y = acc_unit[
    valid_motion,
    1
]


candidate_angles = {

    "X_FORWARD_Y_LEFT": np.degrees(
        np.arctan2(
            phone_y,
            phone_x
        )
    ),

    "X_FORWARD_Y_RIGHT": np.degrees(
        np.arctan2(
            -phone_y,
            phone_x
        )
    ),

    "X_BACKWARD_Y_LEFT": np.degrees(
        np.arctan2(
            phone_y,
            -phone_x
        )
    ),

    "X_BACKWARD_Y_RIGHT": np.degrees(
        np.arctan2(
            -phone_y,
            -phone_x
        )
    )
}


# ============================================================
# 18. FIND BEST HEADING OFFSET
# ============================================================

candidate_results = []


for name, phone_heading_raw in (
    candidate_angles.items()
):

    phone_heading_raw = (
        phone_heading_raw
        + 360
    ) % 360


    difference = angle_difference(
        gps_h,
        phone_heading_raw
    )


    diff_rad = np.radians(
        difference
    )


    offset = np.degrees(
        np.arctan2(
            np.mean(
                np.sin(
                    diff_rad
                )
            ),
            np.mean(
                np.cos(
                    diff_rad
                )
            )
        )
    )


    aligned = (
        phone_heading_raw
        + offset
    ) % 360


    error = np.abs(
        angle_difference(
            aligned,
            gps_h
        )
    )


    candidate_results.append({

        "orientation": name,

        "offset": offset,

        "MAE": np.mean(error),

        "median": np.median(error),

        "P90": np.percentile(
            error,
            90
        )
    })


candidate_results_df = (
    pd.DataFrame(
        candidate_results
    )
    .sort_values(
        "MAE"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 19. RESULTS
# ============================================================

print(
    "\n===== PHONE ORIENTATION TEST ====="
)

print(
    candidate_results_df.to_string(
        index=False
    )
)


best = candidate_results_df.iloc[0]


print(
    "\n===== BEST RESULT ====="
)

print(
    "Orientation:",
    best["orientation"]
)

print(
    "Offset:",
    best["offset"],
    "degrees"
)

print(
    "MAE:",
    best["MAE"],
    "degrees"
)

print(
    "Median:",
    best["median"],
    "degrees"
)

print(
    "P90:",
    best["P90"],
    "degrees"
)


# ============================================================
# 20. SAVE RESULT
# ============================================================

output = df[
    valid_motion
].copy()


output[
    "GNSS_VELOCITY_HEADING"
] = gps_h


best_orientation = (
    best["orientation"]
)


raw_phone_heading = (
    candidate_angles[
        best_orientation
    ]
    + 360
) % 360


aligned_phone_heading = (
    raw_phone_heading
    + best["offset"]
) % 360


output[
    "PHONE_RAW_HEADING"
] = raw_phone_heading


output[
    "PHONE_ALIGNED_HEADING"
] = aligned_phone_heading


output[
    "HEADING_ERROR"
] = angle_difference(
    aligned_phone_heading,
    gps_h
)


OUTPUT_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME"
    r"\data\velocity_alignment_result.csv"
)


output.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\n===== SAVED ====="
)

print(
    "Result saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 21. GRAPH — GNSS HEADING VS DATASET HEADING
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    time[
        valid_heading
    ],
    gnss_heading_smooth[
        valid_heading
    ],
    label="Calculated GNSS Heading"
)

plt.plot(
    time[
        valid_heading
    ],
    gps_orientation[
        valid_heading
    ],
    label="Dataset GPS Orientation",
    alpha=0.7
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "GNSS Velocity Heading Validation"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 22. GRAPH — FINAL ALIGNMENT
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    gps_h,
    label="GNSS Vehicle Heading"
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
    "GNSS Heading vs Aligned Phone Motion"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 23. GRAPH — ERROR
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    output[
        "HEADING_ERROR"
    ].to_numpy()
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
    "Phone → Vehicle Alignment Error"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 24. COMPLETE
# ============================================================

print("\n")
print("====================================================")
print("     VELOCITY ALIGNMENT ANALYSIS COMPLETE")
print("====================================================")

print(
    "\nBest orientation:",
    best_orientation
)

print(
    "Best offset:",
    best["offset"],
    "degrees"
)

print(
    "Final MAE:",
    best["MAE"],
    "degrees"
)

print(
    "Final median:",
    best["median"],
    "degrees"
)

print(
    "Final P90:",
    best["P90"],
    "degrees"
)

print(
    "\n===================================================="
)