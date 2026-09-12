import pandas as pd
import numpy as np

# ============================================================
# PROJECT PRIME
# STEP 7 — GNSS OUTAGE ANALYSIS
# ============================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n========================================")
print("PROJECT PRIME — GNSS OUTAGE ANALYSIS")
print("========================================")

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

time = df["TIME SINCE START (ms)"].to_numpy() / 1000.0

lat = df["GPS LATITUDE (degrees)"].to_numpy()
lon = df["GPS LONGITUDE (degrees)"].to_numpy()

speed = df["GPS SPEED (Kmh)"].to_numpy() / 3.6
heading = df["GPS ORIENTATION (Â°)"].to_numpy()

# ------------------------------------------------------------
# Convert GPS coordinates to local EN coordinates
# ------------------------------------------------------------

R = 6371000.0

lat0 = lat[0]
lon0 = lon[0]

gps_north = (
    np.deg2rad(lat - lat0) * R
)

gps_east = (
    np.deg2rad(lon - lon0)
    * R
    * np.cos(np.deg2rad(lat0))
)

# ------------------------------------------------------------
# GPS velocity components
# ------------------------------------------------------------

heading_rad = np.deg2rad(heading)

gps_vn = speed * np.cos(heading_rad)
gps_ve = speed * np.sin(heading_rad)

# ------------------------------------------------------------
# Linear acceleration
# ------------------------------------------------------------

ax = df["ACCELEROMETER X (m/s²)"].to_numpy()
ay = df["ACCELEROMETER Y (m/s²)"].to_numpy()
az = df["ACCELEROMETER Z (m/s²)"].to_numpy()

gx = df["GRAVITY X (m/s²)"].to_numpy()
gy = df["GRAVITY Y (m/s²)"].to_numpy()
gz = df["GRAVITY Z (m/s²)"].to_numpy()

lin_x = ax - gx
lin_y = ay - gy
lin_z = az - gz

# ------------------------------------------------------------
# Outage durations
# ------------------------------------------------------------

OUTAGES = [5, 10, 20, 30, 60, 120]

# ------------------------------------------------------------
# Choose valid outage starting points
#
# We select points where GPS speed > 5 km/h and enough
# data exists after the starting point.
# ------------------------------------------------------------

valid_starts = np.where(
    (speed > 5 / 3.6)
)[0]

print("\n===== DATA =====")

print("Total samples:", len(df))

print(
    "Duration:",
    time[-1] - time[0],
    "seconds"
)

# ------------------------------------------------------------
# Function: find nearest index at future time
# ------------------------------------------------------------

def future_index(start_idx, duration):

    target_time = time[start_idx] + duration

    idx = np.searchsorted(
        time,
        target_time
    )

    if idx >= len(time):
        return None

    return idx


# ------------------------------------------------------------
# Main outage analysis
# ------------------------------------------------------------

results = []

print("\n========================================")
print("GNSS OUTAGE RESULTS")
print("========================================")

for outage_duration in OUTAGES:

    errors = []
    available_cases = 0

    # Limit number of starting points for reasonable runtime
    step = max(1, len(valid_starts) // 500)

    for start_idx in valid_starts[::step]:

        end_idx = future_index(
            start_idx,
            outage_duration
        )

        if end_idx is None:
            continue

        if end_idx <= start_idx + 1:
            continue

        available_cases += 1

        # ----------------------------------------------------
        # Initial state comes from GPS
        # ----------------------------------------------------

        vx = gps_vn[start_idx]
        vy = gps_ve[start_idx]

        px = gps_north[start_idx]
        py = gps_east[start_idx]

        # ----------------------------------------------------
        # IMU-only propagation during outage
        #
        # NOTE:
        # Sensor-frame alignment is NOT solved yet.
        # We use horizontal acceleration magnitude/sign proxy
        # only to establish the outage experiment.
        # ----------------------------------------------------

        for i in range(
            start_idx + 1,
            end_idx + 1
        ):

            dt = time[i] - time[i - 1]

            if (
                not np.isfinite(dt)
                or dt <= 0
                or dt > 1
            ):
                continue

            # Current sensor acceleration
            a_x = lin_x[i]
            a_y = lin_y[i]

            # Basic integration
            vx += a_x * dt
            vy += a_y * dt

            px += vx * dt
            py += vy * dt

        # ----------------------------------------------------
        # GPS position at outage end
        # ----------------------------------------------------

        true_x = gps_north[end_idx]
        true_y = gps_east[end_idx]

        error = np.sqrt(
            (px - true_x) ** 2 +
            (py - true_y) ** 2
        )

        if np.isfinite(error):
            errors.append(error)

    if len(errors) == 0:
        continue

    errors = np.array(errors)

    result = {
        "duration": outage_duration,
        "cases": len(errors),
        "mean_error": np.mean(errors),
        "median_error": np.median(errors),
        "p90_error": np.percentile(errors, 90),
        "max_error": np.max(errors)
    }

    results.append(result)

    print(
        f"\nOutage: {outage_duration:>3} sec"
    )

    print(
        f"Cases: {len(errors)}"
    )

    print(
        f"Mean error: {np.mean(errors):.2f} m"
    )

    print(
        f"Median error: {np.median(errors):.2f} m"
    )

    print(
        f"P90 error: {np.percentile(errors, 90):.2f} m"
    )

    print(
        f"Maximum error: {np.max(errors):.2f} m"
    )

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

if results:

    results_df = pd.DataFrame(results)

    output_path = (
        r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop"
        r"\PROJECT_PRIME\data\gnss_outage_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n========================================")
    print("RESULTS SAVED")
    print("========================================")

    print(output_path)

print("\n========================================")
print("STEP 7 COMPLETE")
print("========================================")