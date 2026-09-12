import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from prime_core.prime_system import PRIMESystem


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_PATH = os.path.join(
    BASE,
    "data",
    "cleaned_S-S1.csv"
)

SEQ_PATH = os.path.join(
    BASE,
    "data",
    "prime_sequence_dataset_v2.npz"
)

RESULT_PATH = os.path.join(
    BASE,
    "data",
    "prime_final_benchmark.csv"
)

REPORT_PATH = os.path.join(
    BASE,
    "data",
    "prime_final_report.txt"
)

TRAJECTORY_PATH = os.path.join(
    BASE,
    "data",
    "prime_final_trajectory.png"
)

ERROR_PATH = os.path.join(
    BASE,
    "data",
    "prime_final_error.png"
)


print("=" * 80)
print("PROJECT PRIME — STEP 52")
print("FINAL EXPERIMENTAL BENCHMARK")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    encoding="latin1"
)

data = np.load(
    SEQ_PATH
)

X = data["X"]

print("\nDataset")
print("Original samples:", len(df))
print("Sensor sequences:", X.shape)


# ============================================================
# GPS
# ============================================================

lat = pd.to_numeric(
    df["GPS LATITUDE (degrees)"],
    errors="coerce"
).values

lon = pd.to_numeric(
    df["GPS LONGITUDE (degrees)"],
    errors="coerce"
).values

time_ms = pd.to_numeric(
    df["TIME SINCE START (ms)"],
    errors="coerce"
).values


# ============================================================
# FIND GPS FIXES
# ============================================================

fix_indices = []

for i in range(1, len(df)):

    if (
        np.isfinite(lat[i])
        and np.isfinite(lon[i])
        and
        (
            lat[i] != lat[i - 1]
            or
            lon[i] != lon[i - 1]
        )
    ):

        fix_indices.append(i)

fix_indices = np.array(
    fix_indices,
    dtype=int
)

print(
    "GPS fix transitions:",
    len(fix_indices)
)


# ============================================================
# ACTUAL FIX INTERVAL STATISTICS
# ============================================================

fix_times = time_ms[
    fix_indices
]

intervals = np.diff(
    fix_times
) / 1000.0

intervals = intervals[
    np.isfinite(intervals)
]

mean_interval = np.mean(
    intervals
)

median_interval = np.median(
    intervals
)

p90_interval = np.percentile(
    intervals,
    90
)

min_interval = np.min(
    intervals
)

max_interval = np.max(
    intervals
)


print("\nActual GNSS fix interval:")

print(
    f"Mean:   {mean_interval:.3f} s"
)

print(
    f"Median: {median_interval:.3f} s"
)

print(
    f"P90:    {p90_interval:.3f} s"
)

print(
    f"Min:    {min_interval:.3f} s"
)

print(
    f"Max:    {max_interval:.3f} s"
)


# ============================================================
# HAVERSINE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dp = p2 - p1

    dl = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dp / 2) ** 2
        +
        np.cos(p1)
        *
        np.cos(p2)
        *
        np.sin(dl / 2) ** 2
    )

    return (
        2
        *
        R
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# PRIME SYSTEM
# ============================================================

start_idx = fix_indices[0]

prime = PRIMESystem(
    float(lat[start_idx]),
    float(lon[start_idx])
)


# ============================================================
# OUTAGE CONFIGURATION
# ============================================================

# Number of GPS intervals.
#
# We deliberately use the actual dataset interval rather
# than claiming these are exactly 10/20/30/60 seconds.

outage_pattern = [
    1,
    2,
    1,
    3,
    2,
    1,
    6,
    2,
    3,
    1
]


# ============================================================
# STORAGE
# ============================================================

records = []

recovery_records = []

pattern_index = 0

outage_remaining = 0

in_outage = False

outage_start_sequence = None

max_error = 0.0


# ============================================================
# CONTINUOUS BENCHMARK
# ============================================================

for seq_index in range(
    min(
        len(X),
        len(fix_indices) - 1
    )
):

    next_fix = fix_indices[
        seq_index + 1
    ]

    true_lat = float(
        lat[next_fix]
    )

    true_lon = float(
        lon[next_fix]
    )

    # --------------------------------------------------------
    # START OUTAGE
    # --------------------------------------------------------

    if not in_outage:

        if (
            pattern_index
            <
            len(outage_pattern)
        ):

            outage_remaining = (
                outage_pattern[
                    pattern_index
                ]
            )

            pattern_index += 1

            outage_start_sequence = (
                seq_index
            )

            in_outage = True


    # --------------------------------------------------------
    # PRIME DR
    # --------------------------------------------------------

    if in_outage:

        result = prime.process(
            sensor_sequence=X[seq_index],

            satellites=0,

            accuracy=None,

            time_since_update=20.0
        )

        error = haversine(
            result["latitude"],
            result["longitude"],
            true_lat,
            true_lon
        )

        max_error = max(
            max_error,
            error
        )

        records.append({
            "sequence":
                seq_index,

            "mode":
                result["mode"],

            "gnss_state":
                result["gnss_state"],

            "latitude":
                result["latitude"],

            "longitude":
                result["longitude"],

            "true_latitude":
                true_lat,

            "true_longitude":
                true_lon,

            "error_m":
                error,

            "recovered":
                False
        })

        outage_remaining -= 1

        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        if outage_remaining <= 0:

            in_outage = False

            recovery_result = prime.process(
                sensor_sequence=X[seq_index],

                satellites=10,

                accuracy=5.0,

                time_since_update=2.0,

                gnss_latitude=true_lat,

                gnss_longitude=true_lon
            )

            recovery_error = haversine(
                recovery_result["latitude"],
                recovery_result["longitude"],
                true_lat,
                true_lon
            )

            outage_duration = (
                time_ms[next_fix]
                -
                time_ms[
                    fix_indices[
                        outage_start_sequence
                    ]
                ]
            ) / 1000.0

            recovery_records.append({
                "start_sequence":
                    outage_start_sequence,

                "end_sequence":
                    seq_index,

                "duration_seconds":
                    outage_duration,

                "error_before_recovery_m":
                    error,

                "error_after_recovery_m":
                    recovery_error,

                "recovered":
                    recovery_result[
                        "recovered"
                    ]
            })

            records.append({
                "sequence":
                    seq_index,

                "mode":
                    recovery_result[
                        "mode"
                    ],

                "gnss_state":
                    recovery_result[
                        "gnss_state"
                    ],

                "latitude":
                    recovery_result[
                        "latitude"
                    ],

                "longitude":
                    recovery_result[
                        "longitude"
                    ],

                "true_latitude":
                    true_lat,

                "true_longitude":
                    true_lon,

                "error_m":
                    recovery_error,

                "recovered":
                    recovery_result[
                        "recovered"
                    ]
            })

    # --------------------------------------------------------
    # NORMAL GNSS
    # --------------------------------------------------------

    else:

        result = prime.process(
            sensor_sequence=X[seq_index],

            satellites=10,

            accuracy=5.0,

            time_since_update=2.0,

            gnss_latitude=true_lat,

            gnss_longitude=true_lon
        )

        records.append({
            "sequence":
                seq_index,

            "mode":
                result["mode"],

            "gnss_state":
                result["gnss_state"],

            "latitude":
                result["latitude"],

            "longitude":
                result["longitude"],

            "true_latitude":
                true_lat,

            "true_longitude":
                true_lon,

            "error_m":
                0.0,

            "recovered":
                result["recovered"]
        })


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    records
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# PRIME ERROR
# ============================================================

prime_results = results_df[
    results_df["mode"] == "PRIME_DR"
]

errors = prime_results[
    "error_m"
].values


mean_error = np.mean(
    errors
)

median_error = np.median(
    errors
)

p90_error = np.percentile(
    errors,
    90
)

max_prime_error = np.max(
    errors
)


# ============================================================
# RECOVERY
# ============================================================

recovery_df = pd.DataFrame(
    recovery_records
)

successful_recoveries = int(
    recovery_df[
        "recovered"
    ].sum()
)

total_recoveries = len(
    recovery_df
)


# ============================================================
# REPORT
# ============================================================

report = f"""
PROJECT PRIME
FINAL EXPERIMENTAL BENCHMARK
========================================

DATASET
----------------------------------------
Original samples: {len(df)}
Sensor sequences: {len(X)}
Sensor shape: {X.shape[1]} x {X.shape[2]}
GPS fix transitions: {len(fix_indices)}

GNSS FIX INTERVAL
----------------------------------------
Mean:   {mean_interval:.3f} s
Median: {median_interval:.3f} s
P90:    {p90_interval:.3f} s
Minimum: {min_interval:.3f} s
Maximum: {max_interval:.3f} s

CONTINUOUS PRIME TEST
----------------------------------------
PRIME DR samples: {len(prime_results)}
Outage events: {len(recovery_records)}
Recovery events: {total_recoveries}

PRIME POSITION ERROR
----------------------------------------
Mean:   {mean_error:.3f} m
Median: {median_error:.3f} m
P90:    {p90_error:.3f} m
Maximum: {max_prime_error:.3f} m

RECOVERY
----------------------------------------
Successful recoveries:
{successful_recoveries}/{total_recoveries}

Recovery success rate:
{(
    100 * successful_recoveries / total_recoveries
    if total_recoveries
    else 0
):.2f}%

ARCHITECTURE
----------------------------------------
GNSS Monitoring
CNN V2 Displacement Prediction
Position Propagation
GNSS Outage Handling
GNSS Recovery
Automatic Re-anchoring

IMPORTANT LIMITATION
----------------------------------------
This is an offline experimental benchmark on
the available dataset.

It is NOT a production navigation accuracy
validation.

The CNN predicts displacement from sensor
sequences learned from this dataset.

Real-world deployment requires additional
validation with synchronized GNSS/IMU data,
real GNSS-denied experiments, sensor calibration,
and latency/power testing.
"""


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# TRAJECTORY PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    results_df["true_longitude"],
    results_df["true_latitude"],
    label="Reference"
)

prime_mask = (
    results_df["mode"]
    == "PRIME_DR"
)

plt.scatter(
    results_df.loc[
        prime_mask,
        "longitude"
    ],
    results_df.loc[
        prime_mask,
        "latitude"
    ],
    s=12,
    label="PRIME DR"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "PROJECT PRIME — Final Navigation Benchmark"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    TRAJECTORY_PATH,
    dpi=200
)

plt.close()


# ============================================================
# ERROR PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    results_df["sequence"],
    results_df["error_m"]
)

plt.xlabel(
    "Sequence"
)

plt.ylabel(
    "Position Error (m)"
)

plt.title(
    "PROJECT PRIME — Position Error Over Time"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    ERROR_PATH,
    dpi=200
)

plt.close()


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FINAL PRIME BENCHMARK")
print("=" * 80)

print(
    f"\nMean PRIME error:   {mean_error:.3f} m"
)

print(
    f"Median PRIME error: {median_error:.3f} m"
)

print(
    f"P90 PRIME error:    {p90_error:.3f} m"
)

print(
    f"Maximum PRIME error: {max_prime_error:.3f} m"
)

print(
    f"\nRecovery success: "
    f"{successful_recoveries}/{total_recoveries}"
)

print("\nGenerated files:")

print(
    RESULT_PATH
)

print(
    REPORT_PATH
)

print(
    TRAJECTORY_PATH
)

print(
    ERROR_PATH
)

print("\n" + "=" * 80)
print("STEP 52 COMPLETE")
print("=" * 80)