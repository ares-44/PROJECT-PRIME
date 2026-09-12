import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT PRIME — GPS FIX TIMING ANALYSIS")
print("=" * 60)

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(INPUT_PATH)

LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"
TIME = "TIME SINCE START (ms)"
SPEED = "GPS SPEED (Kmh)"

# ============================================================
# IDENTIFY GPS POSITION CHANGES
# ============================================================

lat_change = df[LAT].diff().fillna(0) != 0
lon_change = df[LON].diff().fillna(0) != 0

gps_change = lat_change | lon_change

fix_indices = np.where(gps_change)[0]

print("\n===== GPS FIXES =====")

print("Total samples:", len(df))
print("GPS position changes:", len(fix_indices))

# ============================================================
# TIME BETWEEN GPS FIXES
# ============================================================

fix_times = df[TIME].iloc[fix_indices].values

fix_intervals = np.diff(fix_times) / 1000.0

print("\n===== GPS FIX INTERVAL =====")

print("Mean:", np.mean(fix_intervals), "seconds")
print("Median:", np.median(fix_intervals), "seconds")
print("P10:", np.percentile(fix_intervals, 10), "seconds")
print("P90:", np.percentile(fix_intervals, 90), "seconds")
print("Minimum:", np.min(fix_intervals), "seconds")
print("Maximum:", np.max(fix_intervals), "seconds")

# ============================================================
# DISTRIBUTION
# ============================================================

print("\n===== INTERVAL DISTRIBUTION =====")

for threshold in [
    0.1,
    0.5,
    1,
    2,
    5,
    10,
    20
]:

    count = np.sum(fix_intervals <= threshold)

    print(
        f"Fix intervals <= {threshold} sec:",
        count,
        f"({count / len(fix_intervals) * 100:.2f}%)"
    )

# ============================================================
# FIX-TO-FIX DISTANCE
# ============================================================

R = 6371000.0

lat = np.radians(
    df[LAT].iloc[fix_indices].values
)

lon = np.radians(
    df[LON].iloc[fix_indices].values
)

lat0 = np.mean(lat)

north = lat * R
east = lon * R * np.cos(lat0)

dn = np.diff(north)
de = np.diff(east)

distance = np.sqrt(
    dn**2 + de**2
)

# ============================================================
# FIX-TO-FIX IMPLIED SPEED
# ============================================================

implied_speed = distance / fix_intervals

print("\n===== FIX-TO-FIX MOVEMENT =====")

print(
    "Mean distance:",
    np.mean(distance),
    "m"
)

print(
    "Median distance:",
    np.median(distance),
    "m"
)

print(
    "P90 distance:",
    np.percentile(distance, 90),
    "m"
)

print(
    "Maximum distance:",
    np.max(distance),
    "m"
)

print("\n===== FIX-TO-FIX SPEED =====")

print(
    "Mean:",
    np.mean(implied_speed),
    "m/s"
)

print(
    "Median:",
    np.median(implied_speed),
    "m/s"
)

print(
    "P90:",
    np.percentile(implied_speed, 90),
    "m/s"
)

print(
    "Maximum:",
    np.max(implied_speed),
    "m/s"
)

# ============================================================
# GPS SPEED COMPARISON
# ============================================================

print("\n===== RECORDED GPS SPEED =====")

gps_speed = df[SPEED].iloc[fix_indices].values / 3.6

print(
    "Mean:",
    np.mean(gps_speed),
    "m/s"
)

print(
    "Median:",
    np.median(gps_speed),
    "m/s"
)

print(
    "P90:",
    np.percentile(gps_speed, 90),
    "m/s"
)

print(
    "Maximum:",
    np.max(gps_speed),
    "m/s"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("STEP 14 COMPLETE")
print("=" * 60)