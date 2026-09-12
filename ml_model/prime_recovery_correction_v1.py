import os
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PRIME
# GNSS RECOVERY CORRECTION V1
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

HEALTH_FILE = os.path.join(
    DATA_DIR,
    "gnss_health_monitor_v1_results.csv"
)

AUTO_SWITCH_FILE = os.path.join(
    DATA_DIR,
    "prime_auto_switch_v1_results.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "prime_recovery_correction_v1_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Fraction of the GNSS correction applied at each recovery step.
# 0.25 means 25% of the remaining error is corrected.

CORRECTION_GAIN = 0.25

# Maximum number of correction steps after GNSS returns.

MAX_RECOVERY_STEPS = 4


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PROJECT PRIME — GNSS RECOVERY CORRECTION V1")
print("=" * 70)

print("\nLoading GNSS health data...")

health_df = pd.read_csv(
    HEALTH_FILE
)

print(
    "Health rows:",
    len(health_df)
)


print("\nLoading PRIME auto-switch results...")

auto_df = pd.read_csv(
    AUTO_SWITCH_FILE
)

print(
    "Auto-switch rows:",
    len(auto_df)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "time_seconds",
    "gnss_state",
    "navigation_mode",
    "estimated_north",
    "estimated_east",
    "true_north",
    "true_east"
]

missing = [
    column
    for column in required_columns
    if column not in auto_df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# RECOVERY CORRECTION FUNCTION
# ============================================================

def apply_recovery_correction(
    estimated_north,
    estimated_east,
    gnss_north,
    gnss_east,
    gain
):

    correction_north = (
        gnss_north -
        estimated_north
    )

    correction_east = (
        gnss_east -
        estimated_east
    )

    corrected_north = (
        estimated_north +
        gain *
        correction_north
    )

    corrected_east = (
        estimated_east +
        gain *
        correction_east
    )

    return (
        corrected_north,
        corrected_east
    )


# ============================================================
# SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("RECOVERY CORRECTION SIMULATION")
print("=" * 70)

results = []

previous_state = None
recovery_active = False
recovery_step = 0

estimated_north = 0.0
estimated_east = 0.0


# ============================================================
# PROCESS NAVIGATION STATES
# ============================================================

for i in range(
    len(auto_df)
):

    row = auto_df.iloc[i]

    current_state = row[
        "gnss_state"
    ]

    true_north = float(
        row["true_north"]
    )

    true_east = float(
        row["true_east"]
    )

    # --------------------------------------------------------
    # DETECT GNSS RECOVERY
    # --------------------------------------------------------

    if (
        previous_state == "GNSS_OUTAGE"
        and
        current_state == "GNSS_GOOD"
    ):

        recovery_active = True
        recovery_step = 0

        # Start recovery from PRIME's last position.
        estimated_north = float(
            auto_df.iloc[i - 1][
                "estimated_north"
            ]
        )

        estimated_east = float(
            auto_df.iloc[i - 1][
                "estimated_east"
            ]
        )

        print("\n" + "-" * 70)

        print(
            f"GNSS RECOVERY DETECTED "
            f"at {row['time_seconds']:.2f}s"
        )

        print(
            f"PRIME position: "
            f"N={estimated_north:.2f}, "
            f"E={estimated_east:.2f}"
        )

        print(
            f"GNSS position: "
            f"N={true_north:.2f}, "
            f"E={true_east:.2f}"
        )


    # --------------------------------------------------------
    # RECOVERY MODE
    # --------------------------------------------------------

    if recovery_active:

        recovery_step += 1

        estimated_north, estimated_east = (
            apply_recovery_correction(
                estimated_north,
                estimated_east,
                true_north,
                true_east,
                CORRECTION_GAIN
            )
        )

        correction_error = np.sqrt(

            (
                estimated_north -
                true_north
            ) ** 2

            +

            (
                estimated_east -
                true_east
            ) ** 2
        )

        print(
            f"Recovery step "
            f"{recovery_step}: "
            f"remaining error = "
            f"{correction_error:.2f} m"
        )

        if (
            recovery_step >=
            MAX_RECOVERY_STEPS
        ):

            recovery_active = False

            print(
                ">>> Recovery correction complete"
            )


    # --------------------------------------------------------
    # NORMAL GNSS MODE
    # --------------------------------------------------------

    elif current_state == "GNSS_GOOD":

        estimated_north = true_north
        estimated_east = true_east

        correction_error = 0.0


    # --------------------------------------------------------
    # DEGRADED / OUTAGE
    # --------------------------------------------------------

    else:

        estimated_north = float(
            row["estimated_north"]
        )

        estimated_east = float(
            row["estimated_east"]
        )

        correction_error = np.sqrt(

            (
                estimated_north -
                true_north
            ) ** 2

            +

            (
                estimated_east -
                true_east
            ) ** 2
        )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({

        "time_seconds":
            row["time_seconds"],

        "gnss_state":
            current_state,

        "navigation_mode":
            row["navigation_mode"],

        "recovery_active":
            recovery_active,

        "recovery_step":
            recovery_step
            if recovery_active
            else 0,

        "estimated_north":
            estimated_north,

        "estimated_east":
            estimated_east,

        "true_north":
            true_north,

        "true_east":
            true_east,

        "position_error":
            correction_error
    })


    previous_state = current_state


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# RECOVERY EVENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("RECOVERY PERFORMANCE")
print("=" * 70)

recovery_events = []

previous = None

for i in range(
    len(results_df)
):

    current = results_df.iloc[i][
        "gnss_state"
    ]

    if (
        previous == "GNSS_OUTAGE"
        and
        current == "GNSS_GOOD"
    ):

        recovery_events.append(i)

    previous = current


print(
    "\nRecovery events:",
    len(recovery_events)
)


# ============================================================
# EVENT DETAILS
# ============================================================

event_number = 0

for index in recovery_events:

    event_number += 1

    recovery_time = (
        results_df.iloc[index][
            "time_seconds"
        ]
    )

    recovery_error = (
        results_df.iloc[index][
            "position_error"
        ]
    )

    print(
        f"\nRecovery #{event_number}"
    )

    print(
        f"Time: "
        f"{recovery_time:.2f}s"
    )

    print(
        f"Error after correction: "
        f"{recovery_error:.2f} m"
    )


# ============================================================
# OVERALL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL POSITION ERROR")
print("=" * 70)

print(
    f"Mean error: "
    f"{results_df['position_error'].mean():.3f} m"
)

print(
    f"Median error: "
    f"{results_df['position_error'].median():.3f} m"
)

print(
    f"P90 error: "
    f"{np.percentile(results_df['position_error'], 90):.3f} m"
)

print(
    f"Maximum error: "
    f"{results_df['position_error'].max():.3f} m"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(
    OUTPUT_FILE
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 39 COMPLETE")
print("=" * 70)