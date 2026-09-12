import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT PRIME — REFERENCE TRAJECTORY")
print("=" * 60)

# ============================================================
# PATH
# ============================================================

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_reference_trajectory.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))


# ============================================================
# COLUMN NAMES
# ============================================================

TIME = "TIME SINCE START (ms)"
SPEED = "GPS SPEED (Kmh)"
HEADING = "GPS ORIENTATION (Â°)"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    TIME,
    SPEED,
    HEADING
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR — Missing columns:")
    for col in missing_columns:
        print("-", col)

    print("\nAvailable columns:")
    for col in df.columns:
        print("-", col)

    raise SystemExit


# ============================================================
# TIME
# ============================================================

time = df[TIME].values / 1000.0

dt = np.diff(
    time,
    prepend=time[0]
)

# Protect against invalid/non-positive time intervals
dt[dt <= 0] = 0.1


# ============================================================
# GPS SPEED
# km/h → m/s
# ============================================================

speed = (
    pd.to_numeric(
        df[SPEED],
        errors="coerce"
    )
    .fillna(0)
    .values
    / 3.6
)


# ============================================================
# GPS HEADING
# ============================================================

heading_series = pd.to_numeric(
    df[HEADING],
    errors="coerce"
)

heading_series = (
    heading_series
    .ffill()
    .bfill()
)

heading = heading_series.values

heading_rad = np.radians(heading)


# ============================================================
# VELOCITY COMPONENTS
# ============================================================

# Heading convention:
# 0°   = North
# 90°  = East
# 180° = South
# 270° = West

velocity_north = (
    speed * np.cos(heading_rad)
)

velocity_east = (
    speed * np.sin(heading_rad)
)


# ============================================================
# INTEGRATE VELOCITY
# ============================================================

north_position = np.zeros(len(df))
east_position = np.zeros(len(df))

for i in range(1, len(df)):

    north_position[i] = (
        north_position[i - 1]
        + velocity_north[i] * dt[i]
    )

    east_position[i] = (
        east_position[i - 1]
        + velocity_east[i] * dt[i]
    )


# ============================================================
# ADD REFERENCE TRAJECTORY
# ============================================================

df["REF_NORTH"] = north_position
df["REF_EAST"] = east_position

df["REF_SPEED_MS"] = speed

df["REF_VEL_NORTH"] = velocity_north
df["REF_VEL_EAST"] = velocity_east


# ============================================================
# REFERENCE SPEED STATISTICS
# ============================================================

print("\n===== REFERENCE SPEED =====")

print(
    "Mean:",
    np.mean(speed),
    "m/s"
)

print(
    "Median:",
    np.median(speed),
    "m/s"
)

print(
    "P90:",
    np.percentile(speed, 90),
    "m/s"
)

print(
    "Maximum:",
    np.max(speed),
    "m/s"
)


# ============================================================
# TOTAL TRAVEL DISTANCE
# ============================================================

step_distance = np.sqrt(
    (velocity_north * dt) ** 2
    +
    (velocity_east * dt) ** 2
)

total_distance = np.sum(step_distance)


# ============================================================
# FINAL DISPLACEMENT
# ============================================================

final_north = north_position[-1]
final_east = east_position[-1]

final_displacement = np.sqrt(
    final_north ** 2
    +
    final_east ** 2
)


# ============================================================
# TRAJECTORY RESULTS
# ============================================================

print("\n===== REFERENCE TRAJECTORY =====")

print(
    "Total integrated distance:",
    total_distance,
    "m"
)

print(
    "Final North:",
    final_north,
    "m"
)

print(
    "Final East:",
    final_east,
    "m"
)

print(
    "Final displacement:",
    final_displacement,
    "m"
)


# ============================================================
# SANITY CHECK
# ============================================================

print("\n===== SANITY CHECK =====")

print(
    "Expected recorded GPS distance:",
    np.sum(speed * dt),
    "m"
)

print(
    "Difference:",
    total_distance - np.sum(speed * dt),
    "m"
)

print(
    "Reference trajectory finite:",
    np.all(
        np.isfinite(
            north_position
        )
    )
    and
    np.all(
        np.isfinite(
            east_position
        )
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===== SAVED =====")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("STEP 13 COMPLETE")
print("=" * 60)