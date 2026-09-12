import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT PRIME — GPS JUMP INVESTIGATION")
print("=" * 60)

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(INPUT_PATH)

LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"
SPEED = "GPS SPEED (Kmh)"
ACCURACY = "GPS ACCURACY (m)"
TIME = "TIME SINCE START (ms)"

R = 6371000.0

lat = np.radians(df[LAT].values)
lon = np.radians(df[LON].values)

lat0 = np.mean(lat)

north = lat * R
east = lon * R * np.cos(lat0)

dt = np.diff(
    df[TIME].values,
    prepend=df[TIME].values[0]
) / 1000.0

dt[dt <= 0] = np.nan

dn = np.diff(north, prepend=north[0])
de = np.diff(east, prepend=east[0])

distance = np.sqrt(dn**2 + de**2)

speed = distance / dt

valid = np.isfinite(speed)

# ============================================================
# BASIC GPS MOVEMENT
# ============================================================

print("\n===== GPS MOVEMENT =====")

print("Total samples:", len(df))

print(
    "Samples with movement:",
    np.sum(distance > 0)
)

print(
    "Samples with zero movement:",
    np.sum(distance == 0)
)

print(
    "Percentage zero movement:",
    np.mean(distance == 0) * 100,
    "%"
)

# ============================================================
# JUMP THRESHOLDS
# ============================================================

print("\n===== GPS JUMP COUNTS =====")

for threshold in [1, 2, 5, 10, 20, 50, 100]:

    count = np.sum(speed > threshold)

    print(
        f"> {threshold:3} m/s:",
        count,
        f"({count / len(df) * 100:.3f}%)"
    )

# ============================================================
# TOP 20 GPS JUMPS
# ============================================================

print("\n===== TOP 20 GPS JUMPS =====")

indices = np.argsort(
    np.nan_to_num(speed, nan=-1)
)[-20:][::-1]

for rank, i in enumerate(indices, 1):

    print(
        f"\n#{rank}"
    )

    print(
        "Index:",
        i
    )

    print(
        "Time:",
        df[TIME].iloc[i],
        "ms"
    )

    print(
        "GPS Lat:",
        df[LAT].iloc[i]
    )

    print(
        "GPS Lon:",
        df[LON].iloc[i]
    )

    print(
        "Recorded GPS speed:",
        df[SPEED].iloc[i],
        "km/h"
    )

    print(
        "GPS accuracy:",
        df[ACCURACY].iloc[i],
        "m"
    )

    print(
        "Coordinate jump:",
        distance[i],
        "m"
    )

    print(
        "Implied speed:",
        speed[i],
        "m/s"
    )

# ============================================================
# GPS COORDINATE UNIQUENESS
# ============================================================

print("\n===== GPS UNIQUENESS =====")

unique_lat = df[LAT].nunique()
unique_lon = df[LON].nunique()
unique_pairs = df[[LAT, LON]].drop_duplicates().shape[0]

print("Unique latitude values:", unique_lat)
print("Unique longitude values:", unique_lon)
print("Unique GPS coordinate pairs:", unique_pairs)

# ============================================================
# GPS SPEED DISTRIBUTION
# ============================================================

print("\n===== RECORDED GPS SPEED =====")

print(
    "Mean:",
    df[SPEED].mean(),
    "km/h"
)

print(
    "Median:",
    df[SPEED].median(),
    "km/h"
)

print(
    "P90:",
    df[SPEED].quantile(0.90),
    "km/h"
)

print(
    "Maximum:",
    df[SPEED].max(),
    "km/h"
)

print("\n" + "=" * 60)
print("STEP 12B COMPLETE")
print("=" * 60)