import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT PRIME — REFERENCE TRAJECTORY VALIDATION")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================

ORIGINAL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

REFERENCE_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_reference_trajectory.csv"


# ============================================================
# LOAD
# ============================================================

original = pd.read_csv(ORIGINAL_PATH)
reference = pd.read_csv(REFERENCE_PATH)

print("\n===== DATA =====")
print("Samples:", len(original))


# ============================================================
# GPS COORDINATES
# ============================================================

LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"

R = 6371000.0

lat = np.radians(original[LAT].values)
lon = np.radians(original[LON].values)

lat0 = np.mean(lat)

gps_north = (
    lat - lat[0]
) * R

gps_east = (
    lon - lon[0]
) * R * np.cos(lat0)


# ============================================================
# ACTUAL GPS NET DISPLACEMENT
# ============================================================

gps_final_north = gps_north[-1]
gps_final_east = gps_east[-1]

gps_final_distance = np.sqrt(
    gps_final_north ** 2 +
    gps_final_east ** 2
)


# ============================================================
# REFERENCE TRAJECTORY
# ============================================================

ref_north = reference["REF_NORTH"].values
ref_east = reference["REF_EAST"].values

ref_final_north = ref_north[-1]
ref_final_east = ref_east[-1]

ref_final_distance = np.sqrt(
    ref_final_north ** 2 +
    ref_final_east ** 2
)


# ============================================================
# RESULTS
# ============================================================

print("\n===== ACTUAL GPS NET DISPLACEMENT =====")

print(
    "North:",
    gps_final_north,
    "m"
)

print(
    "East:",
    gps_final_east,
    "m"
)

print(
    "Distance:",
    gps_final_distance,
    "m"
)


print("\n===== REFERENCE TRAJECTORY =====")

print(
    "North:",
    ref_final_north,
    "m"
)

print(
    "East:",
    ref_final_east,
    "m"
)

print(
    "Distance:",
    ref_final_distance,
    "m"
)


# ============================================================
# DIFFERENCE
# ============================================================

north_error = ref_final_north - gps_final_north
east_error = ref_final_east - gps_final_east

distance_error = (
    ref_final_distance -
    gps_final_distance
)


print("\n===== DIFFERENCE =====")

print(
    "North difference:",
    north_error,
    "m"
)

print(
    "East difference:",
    east_error,
    "m"
)

print(
    "Net displacement difference:",
    distance_error,
    "m"
)


# ============================================================
# HEADING TEST
# ============================================================

print("\n===== HEADING RANGE =====")

heading = original[
    "GPS ORIENTATION (Â°)"
].dropna()

print(
    "Minimum:",
    heading.min(),
    "degrees"
)

print(
    "Maximum:",
    heading.max(),
    "degrees"
)

print(
    "Mean:",
    heading.mean(),
    "degrees"
)

print(
    "Median:",
    heading.median(),
    "degrees"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("STEP 13B COMPLETE")
print("=" * 60)