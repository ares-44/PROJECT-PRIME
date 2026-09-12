import os
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PRIME
# GNSS HEALTH MONITOR V1
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DATASET = os.path.join(
    DATA_DIR,
    "cleaned_S-S1.csv"
)

OUTPUT = os.path.join(
    DATA_DIR,
    "gnss_health_monitor_v1_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PROJECT PRIME — GNSS HEALTH MONITOR V1")
print("=" * 70)

df = pd.read_csv(DATASET)

print("\nDataset rows:", len(df))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

SAT_COL = "GPS SATELLITES"
ACC_COL = "GPS ACCURACY (m)"
TIME_COL = "TIME SINCE START (ms)"
LAT_COL = "GPS LATITUDE (degrees)"
LON_COL = "GPS LONGITUDE (degrees)"


required = [
    SAT_COL,
    ACC_COL,
    TIME_COL,
    LAT_COL,
    LON_COL
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# CLEAN SENSOR VALUES
# ============================================================

satellites = pd.to_numeric(
    df[SAT_COL],
    errors="coerce"
)

accuracy = pd.to_numeric(
    df[ACC_COL],
    errors="coerce"
)

time_ms = pd.to_numeric(
    df[TIME_COL],
    errors="coerce"
)

latitude = pd.to_numeric(
    df[LAT_COL],
    errors="coerce"
)

longitude = pd.to_numeric(
    df[LON_COL],
    errors="coerce"
)


# ============================================================
# DETECT GPS POSITION UPDATES
# ============================================================

lat_change = latitude.diff().abs()
lon_change = longitude.diff().abs()

position_changed = (
    (lat_change > 0) |
    (lon_change > 0)
)

position_changed.iloc[0] = True


# ============================================================
# TIME SINCE LAST GPS UPDATE
# ============================================================

last_update_time = None

time_since_update = []

for i in range(len(df)):

    if position_changed.iloc[i]:

        last_update_time = time_ms.iloc[i]

    if last_update_time is None:

        time_since_update.append(
            np.nan
        )

    else:

        elapsed = (
            time_ms.iloc[i] -
            last_update_time
        )

        time_since_update.append(
            max(0.0, elapsed / 1000.0)
        )


time_since_update = np.asarray(
    time_since_update
)


# ============================================================
# GNSS HEALTH CLASSIFICATION
# ============================================================

def classify_gnss(
    sat,
    acc,
    no_update_time
):

    # --------------------------------------------------------
    # OUTAGE
    # --------------------------------------------------------

    if (
        pd.isna(sat)
        or
        pd.isna(acc)
    ):

        return "GNSS_OUTAGE"

    if no_update_time >= 15.0:

        return "GNSS_OUTAGE"

    # --------------------------------------------------------
    # DEGRADED
    # --------------------------------------------------------

    if (
        sat < 4
        or
        acc > 20.0
        or
        no_update_time >= 10.0
    ):

        return "GNSS_DEGRADED"

    # --------------------------------------------------------
    # GOOD
    # --------------------------------------------------------

    return "GNSS_GOOD"


health = []

for i in range(len(df)):

    state = classify_gnss(
        satellites.iloc[i],
        accuracy.iloc[i],
        time_since_update[i]
    )

    health.append(state)


# ============================================================
# BUILD OUTPUT
# ============================================================

result = pd.DataFrame({

    "time_seconds":
        time_ms / 1000.0,

    "gps_satellites":
        satellites,

    "gps_accuracy_m":
        accuracy,

    "position_updated":
        position_changed,

    "time_since_gps_update_s":
        time_since_update,

    "gnss_state":
        health
})


# ============================================================
# STATE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("GNSS HEALTH SUMMARY")
print("=" * 70)

counts = (
    result["gnss_state"]
    .value_counts()
)

for state in [
    "GNSS_GOOD",
    "GNSS_DEGRADED",
    "GNSS_OUTAGE"
]:

    count = int(
        counts.get(
            state,
            0
        )
    )

    percentage = (
        count /
        len(result)
        *
        100
    )

    print(
        f"{state:16s}: "
        f"{count:6d} samples "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# POSITION UPDATE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("POSITION UPDATE STATISTICS")
print("=" * 70)

updates = int(
    position_changed.sum()
)

print(
    "GPS position updates:",
    updates
)

print(
    "Samples without update:",
    len(df) - updates
)

print(
    "Mean time since update:",
    f"{np.nanmean(time_since_update):.3f} s"
)

print(
    "Maximum time since update:",
    f"{np.nanmax(time_since_update):.3f} s"
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)


# ============================================================
# EXAMPLE STATE TRANSITIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST STATE TRANSITIONS")
print("=" * 70)

previous = None
transition_count = 0

for i, state in enumerate(
    result["gnss_state"]
):

    if state != previous:

        print(
            f"{result.iloc[i]['time_seconds']:10.2f}s"
            f"  {previous} → {state}"
        )

        previous = state

        transition_count += 1

        if transition_count >= 20:
            break


print("\n" + "=" * 70)
print("STEP 37 COMPLETE")
print("=" * 70)