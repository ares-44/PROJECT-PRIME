import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROJECT PRIME
# TRAJECTORY VISUALIZATION V1
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "prime_auto_switch_v1_results.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "prime_trajectory_v1.png"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PROJECT PRIME — TRAJECTORY VISUALIZATION V1")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Samples:",
    len(df)
)


# ============================================================
# CREATE TRAJECTORIES
# ============================================================

time = df[
    "time_seconds"
].values

true_north = df[
    "true_north"
].values

true_east = df[
    "true_east"
].values

prime_north = df[
    "estimated_north"
].values

prime_east = df[
    "estimated_east"
].values

state = df[
    "gnss_state"
].values


# ============================================================
# PLOT TRAJECTORIES
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.plot(
    true_east,
    true_north,
    label="GNSS / Reference",
    linewidth=2
)

plt.plot(
    prime_east,
    prime_north,
    label="PRIME CNN V2",
    linewidth=2
)


# ============================================================
# MARK OUTAGE REGIONS
# ============================================================

outage = (
    state ==
    "GNSS_OUTAGE"
)

start = None

for i in range(
    len(outage)
):

    if outage[i] and start is None:

        start = i

    elif (
        not outage[i]
        and start is not None
    ):

        plt.axvspan(
            true_east[start],
            true_east[i - 1],
            alpha=0.15
        )

        start = None


# ============================================================
# START / END
# ============================================================

plt.scatter(
    true_east[0],
    true_north[0],
    s=80,
    label="Start"
)

plt.scatter(
    true_east[-1],
    true_north[-1],
    s=80,
    label="End"
)


# ============================================================
# LABELS
# ============================================================

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "PROJECT PRIME — GNSS vs CNN Dead-Reckoning Trajectory"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.axis(
    "equal"
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=200
)

plt.show()


print("\nSaved:")
print(
    OUTPUT_FILE
)

print("\n" + "=" * 70)
print("STEP 40 COMPLETE")
print("=" * 70)