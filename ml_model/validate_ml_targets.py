import pandas as pd
import numpy as np

print("=" * 50)
print("PROJECT PRIME — ML TARGET VALIDATION")
print("=" * 50)

# ============================================
# LOAD DATA
# ============================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))


# ============================================
# GPS COORDINATES
# ============================================

lat = np.radians(df["GPS LATITUDE (degrees)"].values)
lon = np.radians(df["GPS LONGITUDE (degrees)"].values)

R = 6371000.0

# Convert GPS coordinates to local meters
north = lat * R
east = lon * R * np.cos(np.mean(lat))


# ============================================
# WINDOW
# ============================================

WINDOW = 20

target_distance = []
target_time = []
implied_speed = []
gps_speed = []


# ============================================
# CALCULATE FUTURE DISPLACEMENT
# ============================================

for i in range(len(df) - WINDOW):

    dn = north[i + WINDOW] - north[i]
    de = east[i + WINDOW] - east[i]

    distance = np.sqrt(dn**2 + de**2)

    t = (
        df["TIME SINCE START (ms)"].iloc[i + WINDOW]
        - df["TIME SINCE START (ms)"].iloc[i]
    ) / 1000.0

    if t <= 0:
        continue

    target_distance.append(distance)
    target_time.append(t)
    implied_speed.append(distance / t)

    gps_speed.append(df["GPS SPEED (Kmh)"].iloc[i])


target_distance = np.array(target_distance)
target_time = np.array(target_time)
implied_speed = np.array(implied_speed)
gps_speed = np.array(gps_speed)


# ============================================
# BASIC STATISTICS
# ============================================

print("\n===== TARGET DISTANCE =====")

print("Mean:", np.mean(target_distance))
print("Median:", np.median(target_distance))
print("P90:", np.percentile(target_distance, 90))
print("Maximum:", np.max(target_distance))


# ============================================
# TIME
# ============================================

print("\n===== TIME WINDOW =====")

print("Mean window time:", np.mean(target_time), "seconds")
print("Median window time:", np.median(target_time), "seconds")
print("Minimum:", np.min(target_time))
print("Maximum:", np.max(target_time))


# ============================================
# IMPLIED SPEED
# ============================================

print("\n===== IMPLIED GPS SPEED =====")

print("Mean:", np.mean(implied_speed), "m/s")
print("Median:", np.median(implied_speed), "m/s")
print("P90:", np.percentile(implied_speed, 90), "m/s")
print("Maximum:", np.max(implied_speed), "m/s")

print("\nConverted to km/h:")
print("Mean:", np.mean(implied_speed) * 3.6)
print("Median:", np.median(implied_speed) * 3.6)
print("P90:", np.percentile(implied_speed, 90) * 3.6)
print("Maximum:", np.max(implied_speed) * 3.6)


# ============================================
# GPS RECORDED SPEED
# ============================================

gps_speed_ms = gps_speed / 3.6

print("\n===== RECORDED GPS SPEED =====")

print("Mean:", np.mean(gps_speed_ms), "m/s")
print("Maximum:", np.max(gps_speed_ms), "m/s")


# ============================================
# OUTLIER CHECK
# ============================================

print("\n===== PHYSICAL PLAUSIBILITY =====")

thresholds = [5, 10, 20, 30, 50]

for threshold in thresholds:

    percentage = np.mean(implied_speed > threshold) * 100

    print(
        f"Windows > {threshold} m/s: "
        f"{percentage:.2f}%"
    )


# ============================================
# COMPARE WITH GPS SPEED
# ============================================

ratio = implied_speed / np.maximum(gps_speed_ms, 0.5)

print("\n===== IMPLIED / GPS SPEED RATIO =====")

print("Mean ratio:", np.mean(ratio))
print("Median ratio:", np.median(ratio))
print("P90 ratio:", np.percentile(ratio, 90))

print(
    "Windows where implied speed > "
    "2x GPS speed:",
    np.mean(ratio > 2) * 100,
    "%"
)

print(
    "Windows where implied speed > "
    "5x GPS speed:",
    np.mean(ratio > 5) * 100,
    "%"
)


# ============================================
# FINAL
# ============================================

print("\n" + "=" * 50)
print("STEP 11 COMPLETE")
print("=" * 50)