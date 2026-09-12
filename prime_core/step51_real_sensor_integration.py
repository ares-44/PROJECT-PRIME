import os
import sys
import numpy as np
import pandas as pd

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

SEQ_PATH = os.path.join(
    BASE,
    "data",
    "prime_sequence_dataset_v2.npz"
)

CSV_PATH = os.path.join(
    BASE,
    "data",
    "cleaned_S-S1.csv"
)

RESULT_PATH = os.path.join(
    BASE,
    "data",
    "prime_step51_real_sensor_results.csv"
)


print("=" * 75)
print("PROJECT PRIME — STEP 51")
print("REAL SENSOR → UNIFIED PRIME SYSTEM")
print("=" * 75)


# ============================================================
# LOAD REAL SENSOR SEQUENCES
# ============================================================

data = np.load(SEQ_PATH)

X = data["X"]

print("\nReal sensor dataset:")
print("Sequences:", X.shape[0])
print("Timesteps:", X.shape[1])
print("Channels:", X.shape[2])


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    encoding="latin1"
)

lat_col = "GPS LATITUDE (degrees)"
lon_col = "GPS LONGITUDE (degrees)"

lat = pd.to_numeric(
    df[lat_col],
    errors="coerce"
).values

lon = pd.to_numeric(
    df[lon_col],
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
# INITIALIZE PRIME
# ============================================================

start_lat = float(
    lat[fix_indices[0]]
)

start_lon = float(
    lon[fix_indices[0]]
)

prime = PRIMESystem(
    start_lat,
    start_lon
)

print("\nInitial position:")

print(
    prime.position_engine.get_position()
)


# ============================================================
# HELPER
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
# SIMULATION
# ============================================================

results = []

outage_events = []

recovery_events = []

# Deterministic outage pattern.
#
# Each value = number of GPS intervals
# for which GNSS is artificially unavailable.

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

pattern_index = 0

outage_remaining = 0

in_outage = False

max_error = 0.0


# ============================================================
# PROCESS REAL SEQUENCES
# ============================================================

for seq_index in range(
    min(
        len(X),
        len(fix_indices) - 1
    )
):

    current_fix = fix_indices[
        seq_index
    ]

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

            in_outage = True

            outage_events.append({
                "sequence":
                    seq_index,
                "duration_intervals":
                    outage_remaining
            })

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

        results.append({
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
                result["recovered"],

            "recovery_count":
                result["recovery_count"]
        })

        outage_remaining -= 1

        # ----------------------------------------------------
        # GNSS RECOVERY
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

            recovery_events.append({
                "sequence":
                    seq_index,

                "error_before_recovery_m":
                    error,

                "error_after_recovery_m":
                    haversine(
                        recovery_result[
                            "latitude"
                        ],
                        recovery_result[
                            "longitude"
                        ],
                        true_lat,
                        true_lon
                    ),

                "recovered":
                    recovery_result[
                        "recovered"
                    ]
            })

            results.append({
                "sequence":
                    seq_index,

                "mode":
                    recovery_result["mode"],

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
                    0.0,

                "recovered":
                    recovery_result[
                        "recovered"
                    ],

                "recovery_count":
                    recovery_result[
                        "recovery_count"
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

        results.append({
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
                result["recovered"],

            "recovery_count":
                result["recovery_count"]
        })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

prime_results = results_df[
    results_df["mode"]
    == "PRIME_DR"
]

errors = prime_results[
    "error_m"
].values


print("\n" + "=" * 75)
print("STEP 51 RESULTS")
print("=" * 75)

print(
    "\nSequences processed:",
    len(results_df)
)

print(
    "PRIME DR samples:",
    len(prime_results)
)

print(
    "Outage events:",
    len(outage_events)
)

print(
    "Recovery events:",
    len(recovery_events)
)

if len(errors) > 0:

    print(
        "\nREAL SENSOR PRIME ERROR:"
    )

    print(
        f"Mean:   {np.mean(errors):.3f} m"
    )

    print(
        f"Median: {np.median(errors):.3f} m"
    )

    print(
        f"P90:    {np.percentile(errors, 90):.3f} m"
    )

    print(
        f"Max:    {np.max(errors):.3f} m"
    )


# ============================================================
# RECOVERY RESULTS
# ============================================================

print(
    "\nRECOVERY VALIDATION"
)

for i, event in enumerate(
    recovery_events,
    start=1
):

    print(
        f"Recovery #{i}: "
        f"{event['error_before_recovery_m']:.2f} m"
        f" → "
        f"{event['error_after_recovery_m']:.2f} m "
        f"| success="
        f"{event['recovered']}"
    )


successful = sum(
    event["recovered"]
    for event in recovery_events
)

print(
    "\nSuccessful recoveries:",
    successful,
    "/",
    len(recovery_events)
)


# ============================================================
# FINAL
# ============================================================

print("\nSaved:")

print(
    RESULT_PATH
)

print("\n" + "=" * 75)

print(
    "STEP 51 COMPLETE"
)

print(
    "REAL SENSOR DATA → PRIME SYSTEM → RECOVERY"
)

print("=" * 75)