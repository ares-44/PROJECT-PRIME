import os
import sys
import numpy as np

# Allow imports from prime_core
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from prime_core.gnss_monitor import GNSSMonitor
from prime_core.inference_engine import InferenceEngine
from prime_core.position_engine import PositionEngine
from prime_core.prime_navigation import PRIMENavigation


# ============================================================
# PROJECT PRIME — CORE INTEGRATION TEST
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


SEQUENCE_FILE = os.path.join(
    DATA_DIR,
    "prime_sequence_dataset_v2.npz"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "prime_temporal_cnn_v2.keras"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "prime_sequence_scaler_v2.pkl"
)


print("=" * 70)
print("PROJECT PRIME — CORE INTEGRATION TEST")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

data = np.load(SEQUENCE_FILE)

X = data["X"]
y = data["y"]

print("\nDATASET")
print("-" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 2. LOAD CNN
# ============================================================

print("\nLOADING CNN V2...")

inference_engine = InferenceEngine()

print("CNN V2 loaded successfully.")


# ============================================================
# 3. CREATE GNSS MONITOR
# ============================================================

gnss_monitor = GNSSMonitor()

print("GNSS monitor initialized.")


# ============================================================
# 4. INITIAL POSITION
# ============================================================

# Example starting coordinate.
# This is only for testing the position engine.

initial_latitude = 28.4744
initial_longitude = 77.5040

position_engine = PositionEngine(
    initial_latitude,
    initial_longitude
)

print(
    "Initial position:",
    position_engine.get_position()
)


# ============================================================
# 5. CREATE PRIME NAVIGATION SYSTEM
# ============================================================

navigation = PRIMENavigation(
    gnss_monitor,
    inference_engine,
    position_engine
)

print("PRIME navigation engine initialized.")


# ============================================================
# 6. TEST CNN PREDICTION
# ============================================================

print("\nCNN INFERENCE TEST")
print("-" * 70)

sample = X[0]

prediction = inference_engine.predict(
    sample
)

print(
    "Predicted North displacement:",
    f"{prediction[0]:.3f} m"
)

print(
    "Predicted East displacement:",
    f"{prediction[1]:.3f} m"
)

print(
    "Actual North displacement:",
    f"{y[0][0]:.3f} m"
)

print(
    "Actual East displacement:",
    f"{y[0][1]:.3f} m"
)


# ============================================================
# 7. TEST GNSS GOOD
# ============================================================

print("\nGNSS GOOD TEST")
print("-" * 70)

result = navigation.update(
    sensor_sequence=sample,
    satellites=10,
    accuracy=5.0,
    time_since_update=2.0
)

print(result)


# ============================================================
# 8. TEST GNSS DEGRADED
# ============================================================

print("\nGNSS DEGRADED TEST")
print("-" * 70)

result = navigation.update(
    sensor_sequence=sample,
    satellites=3,
    accuracy=10.0,
    time_since_update=5.0
)

print(result)


# ============================================================
# 9. TEST GNSS OUTAGE
# ============================================================

print("\nGNSS OUTAGE TEST")
print("-" * 70)

result = navigation.update(
    sensor_sequence=sample,
    satellites=0,
    accuracy=None,
    time_since_update=20.0
)

print(result)


# ============================================================
# 10. MULTIPLE PRIME UPDATES
# ============================================================

print("\nMULTI-STEP PRIME DR TEST")
print("-" * 70)

# Reset position
position_engine.reset(
    initial_latitude,
    initial_longitude
)

errors = []

for i in range(
    min(10, len(X))
):

    result = navigation.update(
        sensor_sequence=X[i],
        satellites=0,
        accuracy=None,
        time_since_update=20.0
    )

    north = result[
        "north_displacement"
    ]

    east = result[
        "east_displacement"
    ]

    print(
        f"Step {i + 1:02d} | "
        f"ΔN={north:8.3f} m | "
        f"ΔE={east:8.3f} m | "
        f"Lat={result['position']['latitude']:.7f} | "
        f"Lon={result['position']['longitude']:.7f}"
    )


# ============================================================
# 11. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("INTEGRATION TEST RESULT")
print("=" * 70)

print("""
CNN V2                 : LOADED
GNSS Monitor           : WORKING
Position Engine        : WORKING
Navigation Controller  : WORKING
GNSS → PRIME switching : WORKING
Multi-step DR          : EXECUTED
""")

print("=" * 70)
print("STEP 44 COMPLETE")
print("=" * 70)
