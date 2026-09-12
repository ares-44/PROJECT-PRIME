import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT PRIME — DEAD RECKONING BASELINE V2")
print("=" * 60)

# ============================================================
# PATH
# ============================================================

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))


# ============================================================
# COLUMNS
# ============================================================

TIME = "TIME SINCE START (ms)"
LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"

ACC_X = "ACCELEROMETER X (m/s²)"
ACC_Y = "ACCELEROMETER Y (m/s²)"
ACC_Z = "ACCELEROMETER Z (m/s²)"

GRAV_X = "GRAVITY X (m/s²)"
GRAV_Y = "GRAVITY Y (m/s²)"
GRAV_Z = "GRAVITY Z (m/s²)"


# ============================================================
# TIME
# ============================================================

time = (
    df[TIME].values / 1000.0
)

dt = np.diff(
    time,
    prepend=time[0]
)

dt[dt <= 0] = 0.1


# ============================================================
# GPS → LOCAL METERS
# ============================================================

R = 6371000.0

lat = np.radians(
    df[LAT].values
)

lon = np.radians(
    df[LON].values
)

lat0 = np.mean(lat)

gps_north = lat * R
gps_east = lon * R * np.cos(lat0)


# ============================================================
# LINEAR ACCELERATION
# ============================================================

acc_x = (
    df[ACC_X].fillna(0).values
)

acc_y = (
    df[ACC_Y].fillna(0).values
)

acc_z = (
    df[ACC_Z].fillna(0).values
)

grav_x = (
    df[GRAV_X].fillna(0).values
)

grav_y = (
    df[GRAV_Y].fillna(0).values
)

grav_z = (
    df[GRAV_Z].fillna(0).values
)


# Remove gravity
lin_x = acc_x - grav_x
lin_y = acc_y - grav_y
lin_z = acc_z - grav_z


# ============================================================
# IMPORTANT:
# GPS FIXES
# ============================================================

gps_changed = (
    df[LAT].diff().fillna(0).ne(0)
    |
    df[LON].diff().fillna(0).ne(0)
)

fix_indices = np.where(
    gps_changed.values
)[0]

print("\n===== GPS FIXES =====")
print("GPS fixes:", len(fix_indices))


# ============================================================
# DRIFT RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH GPS INTERVAL
# ============================================================

for j in range(len(fix_indices) - 1):

    start = fix_indices[j]
    end = fix_indices[j + 1]

    interval = (
        time[end] -
        time[start]
    )

    # Ignore abnormal GPS gaps
    if interval < 5 or interval > 15:
        continue


    # --------------------------------------------------------
    # TRUE GPS DISPLACEMENT
    # --------------------------------------------------------

    true_north = (
        gps_north[end] -
        gps_north[start]
    )

    true_east = (
        gps_east[end] -
        gps_east[start]
    )

    true_distance = np.sqrt(
        true_north ** 2 +
        true_east ** 2
    )


    # --------------------------------------------------------
    # IMU INTEGRATION
    # --------------------------------------------------------

    velocity_x = 0.0
    velocity_y = 0.0

    position_x = 0.0
    position_y = 0.0


    for i in range(
        start + 1,
        end + 1
    ):

        # Horizontal sensor-frame acceleration
        ax = lin_x[i]
        ay = lin_y[i]

        velocity_x += ax * dt[i]
        velocity_y += ay * dt[i]

        position_x += velocity_x * dt[i]
        position_y += velocity_y * dt[i]


    imu_distance = np.sqrt(
        position_x ** 2 +
        position_y ** 2
    )


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    error = np.sqrt(
        (position_x - true_north) ** 2
        +
        (position_y - true_east) ** 2
    )


    results.append({
        "start_index": start,
        "end_index": end,
        "interval": interval,
        "true_north": true_north,
        "true_east": true_east,
        "true_distance": true_distance,
        "imu_north": position_x,
        "imu_east": position_y,
        "imu_distance": imu_distance,
        "error": error
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n===== DR RESULTS =====")

print(
    "Valid intervals:",
    len(results_df)
)

if len(results_df) > 0:

    error = results_df["error"].values

    print(
        "\nMean error:",
        np.mean(error),
        "m"
    )

    print(
        "Median error:",
        np.median(error),
        "m"
    )

    print(
        "P90 error:",
        np.percentile(error, 90),
        "m"
    )

    print(
        "Maximum error:",
        np.max(error),
        "m"
    )


    # ========================================================
    # TRUE DISTANCE
    # ========================================================

    true_distance = (
        results_df["true_distance"]
        .values
    )

    imu_distance = (
        results_df["imu_distance"]
        .values
    )

    print(
        "\nMean GPS displacement:",
        np.mean(true_distance),
        "m"
    )

    print(
        "Mean IMU displacement:",
        np.mean(imu_distance),
        "m"
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\dr_baseline_v2_results.csv"

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nResults saved:",
        OUTPUT_PATH
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("STEP 16 COMPLETE")
print("=" * 60)