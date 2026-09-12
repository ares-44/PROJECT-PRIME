import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PRIME — PERFORMANCE REPORT V1
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


# ============================================================
# FILES
# ============================================================

BENCHMARK_FILE = os.path.join(
    DATA_DIR,
    "unified_outage_benchmark_v1.csv"
)

AUTO_FILE = os.path.join(
    DATA_DIR,
    "prime_auto_switch_v1_results.csv"
)

RECOVERY_FILE = os.path.join(
    DATA_DIR,
    "prime_recovery_correction_v1_results.csv"
)

OUTPUT_CSV = os.path.join(
    DATA_DIR,
    "prime_performance_summary_v1.csv"
)

ERROR_PLOT = os.path.join(
    DATA_DIR,
    "prime_error_over_time_v1.png"
)

MODEL_PLOT = os.path.join(
    DATA_DIR,
    "prime_model_comparison_v1.png"
)

STATE_PLOT = os.path.join(
    DATA_DIR,
    "prime_gnss_state_distribution_v1.png"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PROJECT PRIME — PERFORMANCE REPORT V1")
print("=" * 70)

benchmark = pd.read_csv(
    BENCHMARK_FILE
)

auto = pd.read_csv(
    AUTO_FILE
)

recovery = pd.read_csv(
    RECOVERY_FILE
)

print("\nFiles loaded successfully.")


# ============================================================
# CNN V2 RESULTS
# ============================================================

cnn = benchmark[
    benchmark["model"] == "CNN V2"
].copy()

cnn = cnn.sort_values(
    "intervals"
)


# ============================================================
# PRINT BENCHMARK
# ============================================================

print("\n" + "=" * 70)
print("CNN V2 OUTAGE PERFORMANCE")
print("=" * 70)

print(
    cnn[
        [
            "outage",
            "trials",
            "mean",
            "median",
            "p90",
            "max"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# AUTO SWITCH STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("AUTO-SWITCH PERFORMANCE")
print("=" * 70)

mode_counts = (
    auto["navigation_mode"]
    .value_counts()
)

print(
    mode_counts.to_string()
)

prime_activations = 0

previous = False

for active in auto[
    "prime_active"
]:

    if active and not previous:
        prime_activations += 1

    previous = active

print(
    "\nPRIME activation events:",
    prime_activations
)

print(
    "Maximum position error:",
    f"{auto['position_error'].max():.3f} m"
)

print(
    "Mean position error:",
    f"{auto['position_error'].mean():.3f} m"
)


# ============================================================
# RECOVERY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RECOVERY PERFORMANCE")
print("=" * 70)

recovery_events = recovery[
    recovery["recovery_active"] == True
]

print(
    "Recovery samples:",
    len(recovery_events)
)

print(
    "Maximum recovery error:",
    f"{recovery['position_error'].max():.3f} m"
)


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary_rows = []

for _, row in cnn.iterrows():

    summary_rows.append({

        "model":
            "CNN V2",

        "outage":
            row["outage"],

        "trials":
            row["trials"],

        "mean_error_m":
            row["mean"],

        "median_error_m":
            row["median"],

        "p90_error_m":
            row["p90"],

        "maximum_error_m":
            row["max"]
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PLOT 1 — POSITION ERROR OVER TIME
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    auto["time_seconds"],
    auto["position_error"],
    linewidth=1.5
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Position error (m)"
)

plt.title(
    "PROJECT PRIME — Position Error Over Time"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    ERROR_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2 — MODEL COMPARISON
# ============================================================

models = [
    "RF V2",
    "CNN V2",
    "Hybrid CNN V1"
]

outage_order = [
    "10 sec",
    "20 sec",
    "30 sec",
    "60 sec"
]

plt.figure(
    figsize=(12, 7)
)

for model_name in models:

    subset = benchmark[
        benchmark["model"] ==
        model_name
    ].copy()

    values = []

    for outage in outage_order:

        match = subset[
            subset["outage"] ==
            outage
        ]

        if len(match) > 0:
            values.append(
                match.iloc[0]["mean"]
            )
        else:
            values.append(
                np.nan
            )

    plt.plot(
        outage_order,
        values,
        marker="o",
        label=model_name
    )


plt.xlabel(
    "GNSS outage duration"
)

plt.ylabel(
    "Mean position error (m)"
)

plt.title(
    "PROJECT PRIME — Model Comparison"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    MODEL_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3 — GNSS STATE DISTRIBUTION
# ============================================================

state_counts = (
    auto["gnss_state"]
    .value_counts()
)

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    state_counts.index,
    state_counts.values
)

plt.xlabel(
    "GNSS state"
)

plt.ylabel(
    "Number of samples"
)

plt.title(
    "PROJECT PRIME — GNSS State Distribution"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    STATE_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT PRIME — FINAL PERFORMANCE SUMMARY")
print("=" * 70)

best_row = cnn.loc[
    cnn["mean"].idxmin()
]

print(
    "\nCurrent ML engine: CNN V2"
)

print(
    f"10 sec mean error : "
    f"{cnn.iloc[0]['mean']:.2f} m"
)

print(
    f"20 sec mean error : "
    f"{cnn.iloc[1]['mean']:.2f} m"
)

print(
    f"30 sec mean error : "
    f"{cnn.iloc[2]['mean']:.2f} m"
)

print(
    f"60 sec mean error : "
    f"{cnn.iloc[3]['mean']:.2f} m"
)

print(
    f"\nAuto-switch events: "
    f"{prime_activations}"
)

print(
    f"Auto-switch maximum error: "
    f"{auto['position_error'].max():.2f} m"
)


# ============================================================
# SAVED FILES
# ============================================================

print("\n" + "=" * 70)
print("GENERATED FILES")
print("=" * 70)

print(
    "\nSummary CSV:"
)

print(
    OUTPUT_CSV
)

print(
    "\nError plot:"
)

print(
    ERROR_PLOT
)

print(
    "\nModel comparison:"
)

print(
    MODEL_PLOT
)

print(
    "\nGNSS state distribution:"
)

print(
    STATE_PLOT
)


print("\n" + "=" * 70)
print("STEP 41 COMPLETE")
print("=" * 70)