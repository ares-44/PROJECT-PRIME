# ============================================================
# PROJECT PRIME
# Phone Sensor → Vehicle Heading Alignment
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

DATA_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME"
    r"\data\cleaned_S-S1.csv"
)

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("\n===== DATA LOADED =====")
print("Total samples:", len(df))


# ------------------------------------------------------------
# 2. CREATE TIME
# ------------------------------------------------------------

df["time_seconds"] = (
    df["TIME SINCE START (ms)"] / 1000.0
)


# ------------------------------------------------------------
# 3. SELECT REQUIRED SENSOR DATA
# ------------------------------------------------------------

gravity = df[
    [
        "GRAVITY X (m/s²)",
        "GRAVITY Y (m/s²)",
        "GRAVITY Z (m/s²)"
    ]
].to_numpy()

magnetic = df[
    [
        "MAGNETIC FIELD X (Î¼T)",
        "MAGNETIC FIELD Y (Î¼T)",
        "MAGNETIC FIELD Z (Î¼T)"
    ]
].to_numpy()

gps_heading = df[
    "GPS ORIENTATION (Â°)"
].to_numpy()

gps_speed = df[
    "GPS SPEED (Kmh)"
].to_numpy()


# ------------------------------------------------------------
# 4. NORMALIZE GRAVITY VECTOR
# ------------------------------------------------------------

gravity_norm = np.linalg.norm(
    gravity,
    axis=1,
    keepdims=True
)

gravity_norm[gravity_norm == 0] = 1

gravity_unit = (
    gravity / gravity_norm
)


# ------------------------------------------------------------
# 5. REMOVE VERTICAL MAGNETIC COMPONENT
#
# This projects the magnetic-field vector onto the
# horizontal plane defined by gravity.
# ------------------------------------------------------------

mag_vertical_component = (
    np.sum(
        magnetic * gravity_unit,
        axis=1,
        keepdims=True
    )
    * gravity_unit
)

mag_horizontal = (
    magnetic - mag_vertical_component
)


# ------------------------------------------------------------
# 6. NORMALIZE HORIZONTAL MAGNETIC VECTOR
# ------------------------------------------------------------

mag_horizontal_norm = np.linalg.norm(
    mag_horizontal,
    axis=1,
    keepdims=True
)

mag_horizontal_norm[
    mag_horizontal_norm == 0
] = 1

mag_horizontal_unit = (
    mag_horizontal / mag_horizontal_norm
)


# ------------------------------------------------------------
# 7. CALCULATE MAGNETIC HEADING
#
# atan2(Y, X) gives the horizontal direction.
# ------------------------------------------------------------

mag_heading = np.degrees(
    np.arctan2(
        mag_horizontal_unit[:, 1],
        mag_horizontal_unit[:, 0]
    )
)

mag_heading = (
    mag_heading + 360
) % 360


# ------------------------------------------------------------
# 8. CREATE RELIABLE GNSS REFERENCE
#
# Use only moving vehicles.
# ------------------------------------------------------------

moving = gps_speed > 5

valid_indices = (
    moving
    & np.isfinite(gps_heading)
    & np.isfinite(mag_heading)
)


valid = df.loc[
    valid_indices
].copy()


gps_reference = gps_heading[
    valid_indices
]

mag_reference = mag_heading[
    valid_indices
]


print("\n===== VALID DATA =====")
print("Reliable moving samples:", len(valid))


# ------------------------------------------------------------
# 9. ANGLE DIFFERENCE FUNCTION
# ------------------------------------------------------------

def angle_difference(a, b):
    """
    Returns shortest angular difference in degrees.
    Result is in range [-180, 180].
    """

    return (
        (a - b + 180) % 360
    ) - 180


# ------------------------------------------------------------
# 10. TEST DIFFERENT SENSOR TRANSFORMATIONS
#
# Phone coordinate systems can have different conventions.
# We therefore test:
#
# MAG
# -MAG
# MAG ± 90
# -MAG ± 90
# MAG + 180
# -MAG + 180
# ------------------------------------------------------------

transformations = {

    "MAG": mag_reference,

    "-MAG": (
        mag_reference + 180
    ) % 360,

    "MAG + 90": (
        mag_reference + 90
    ) % 360,

    "MAG - 90": (
        mag_reference - 90
    ) % 360,

    "MAG + 180": (
        mag_reference + 180
    ) % 360,

    "-MAG + 90": (
        mag_reference + 180 + 90
    ) % 360,

    "-MAG - 90": (
        mag_reference + 180 - 90
    ) % 360,

    "-MAG + 180": (
        mag_reference + 180 + 180
    ) % 360,
}


# ------------------------------------------------------------
# 11. FIND BEST TRANSFORMATION
#
# Instead of normal mean, use circular statistics.
# ------------------------------------------------------------

results = []


for name, heading in transformations.items():

    difference = angle_difference(
        gps_reference,
        heading
    )

    # Circular offset
    radians = np.radians(
        difference
    )

    circular_offset = np.degrees(
        np.arctan2(
            np.mean(np.sin(radians)),
            np.mean(np.cos(radians))
        )
    )

    # Apply offset
    aligned_heading = (
        heading + circular_offset
    ) % 360

    error = angle_difference(
        aligned_heading,
        gps_reference
    )

    abs_error = np.abs(error)

    results.append({

        "transformation": name,

        "offset": circular_offset,

        "MAE": np.mean(abs_error),

        "median_error": np.median(
            abs_error
        ),

        "p90_error": np.percentile(
            abs_error,
            90
        )
    })


results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# 12. SORT RESULTS
# ------------------------------------------------------------

results_df = results_df.sort_values(
    "MAE"
).reset_index(drop=True)


print("\n===== TRANSFORMATION TEST =====")

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 13. SELECT BEST TRANSFORMATION
# ------------------------------------------------------------

best = results_df.iloc[0]

best_name = best[
    "transformation"
]

best_offset = best[
    "offset"
]


print("\n===== BEST TRANSFORMATION =====")

print(
    "Transformation:",
    best_name
)

print(
    "Estimated offset:",
    best_offset,
    "degrees"
)

print(
    "MAE:",
    best["MAE"],
    "degrees"
)

print(
    "Median error:",
    best["median_error"],
    "degrees"
)

print(
    "90th percentile:",
    best["p90_error"],
    "degrees"
)


# ------------------------------------------------------------
# 14. APPLY BEST TRANSFORMATION
# ------------------------------------------------------------

best_heading = transformations[
    best_name
]


aligned_heading = (
    best_heading + best_offset
) % 360


valid[
    "ALIGNED_PHONE_HEADING"
] = aligned_heading


# ------------------------------------------------------------
# 15. CALCULATE FINAL HEADING ERROR
# ------------------------------------------------------------

valid[
    "HEADING_ERROR"
] = angle_difference(
    valid[
        "ALIGNED_PHONE_HEADING"
    ].to_numpy(),

    valid[
        "GPS ORIENTATION (Â°)"
    ].to_numpy()
)


# ------------------------------------------------------------
# 16. FINAL ALIGNMENT CHECK
# ------------------------------------------------------------

final_error = np.abs(
    valid[
        "HEADING_ERROR"
    ].to_numpy()
)


print("\n===== FINAL ALIGNMENT CHECK =====")

print(
    "Transformation used:",
    best_name
)

print(
    "Offset used:",
    best_offset,
    "degrees"
)

print(
    "Calculated MAE:",
    np.mean(final_error),
    "degrees"
)

print(
    "Calculated median error:",
    np.median(final_error),
    "degrees"
)

print(
    "Calculated 90th percentile:",
    np.percentile(
        final_error,
        90
    ),
    "degrees"
)


# ------------------------------------------------------------
# 17. SAVE ALIGNMENT RESULT
# ------------------------------------------------------------

OUTPUT_PATH = (
    r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME"
    r"\data\rotation_alignment_result.csv"
)


valid.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n===== SAVED =====")

print(
    "Alignment result saved to:"
)

print(
    OUTPUT_PATH
)


# ------------------------------------------------------------
# 18. GRAPH 1
# GNSS HEADING vs ALIGNED PHONE HEADING
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    valid["time_seconds"],
    valid["GPS ORIENTATION (Â°)"],
    label="GNSS Heading"
)

plt.plot(
    valid["time_seconds"],
    valid["ALIGNED_PHONE_HEADING"],
    label="Aligned Phone Heading",
    alpha=0.7
)

plt.xlabel(
    "Time (seconds)"
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


# ------------------------------------------------------------
# 19. GRAPH 2
# HEADING ERROR
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    valid["time_seconds"],
    valid["HEADING_ERROR"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Heading Error (degrees)"
)

plt.title(
    "Phone-to-GNSS Heading Alignment Error"
)

plt.grid()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. GRAPH 3
# MAGNETIC HEADING vs GNSS
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    valid["time_seconds"],
    gps_reference,
    label="GNSS Heading"
)

plt.plot(
    valid["time_seconds"],
    mag_reference,
    label="Raw Magnetic Heading",
    alpha=0.7
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "GNSS Heading vs Raw Magnetic Heading"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. COMPLETE
# ------------------------------------------------------------

print("\n========================================")
print("ROTATION ALIGNMENT ANALYSIS COMPLETE")
print("========================================")