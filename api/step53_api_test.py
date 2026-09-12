import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    BASE_DIR
)

from api.prime_api import PRIMEAPI


print("=" * 75)
print("PROJECT PRIME — STEP 53")
print("PRIME API LAYER TEST")
print("=" * 75)


# ============================================================
# INITIALIZE
# ============================================================

prime = PRIMEAPI(
    initial_latitude=52.4016600,
    initial_longitude=-1.5052900
)

print("\nAPI initialized.")

print(
    "Initial position:",
    prime.get_position()
)


# ============================================================
# REAL SENSOR DATA
# ============================================================

sequence_path = os.path.join(
    BASE_DIR,
    "data",
    "prime_sequence_dataset_v2.npz"
)

data = np.load(
    sequence_path
)

X = data["X"]

print(
    "\nLoaded sensor data:",
    X.shape
)


# ============================================================
# TEST 1 — NORMAL GNSS
# ============================================================

print("\n" + "-" * 75)
print("TEST 1 — GNSS NORMAL")
print("-" * 75)

result = prime.process_navigation(

    sensor_sequence=X[0],

    satellites=10,

    accuracy=5.0,

    time_since_update=2.0,

    gnss_latitude=52.4016600,

    gnss_longitude=-1.5052900
)

print(result)

assert result["success"] is True

assert (
    result["navigation"]["mode"]
    == "GNSS"
)

print("NORMAL TEST: PASSED")


# ============================================================
# TEST 2 — GNSS OUTAGE
# ============================================================

print("\n" + "-" * 75)
print("TEST 2 — GNSS OUTAGE")
print("-" * 75)

result = prime.process_navigation(

    sensor_sequence=X[1],

    satellites=0,

    accuracy=None,

    time_since_update=20.0
)

print(result)

assert result["success"] is True

assert (
    result["navigation"]["mode"]
    == "PRIME_DR"
)

assert (
    result["navigation"]["gnss_state"]
    == "OUTAGE"
)

print("OUTAGE TEST: PASSED")


# ============================================================
# TEST 3 — CONTINUOUS PRIME
# ============================================================

print("\n" + "-" * 75)
print("TEST 3 — CONTINUOUS PRIME DR")
print("-" * 75)

result = prime.process_navigation(

    sensor_sequence=X[2],

    satellites=0,

    accuracy=None,

    time_since_update=20.0
)

print(result)

assert (
    result["navigation"]["mode"]
    == "PRIME_DR"
)

print("CONTINUOUS DR TEST: PASSED")


# ============================================================
# TEST 4 — RECOVERY
# ============================================================

print("\n" + "-" * 75)
print("TEST 4 — GNSS RECOVERY")
print("-" * 75)

result = prime.process_navigation(

    sensor_sequence=X[3],

    satellites=10,

    accuracy=5.0,

    time_since_update=2.0,

    gnss_latitude=52.4050000,

    gnss_longitude=-1.5100000
)

print(result)

assert result["success"] is True

assert (
    result["navigation"]["mode"]
    == "GNSS"
)

assert (
    result["recovery"]["recovered"]
    is True
)

assert (
    result["recovery"]["recovery_count"]
    == 1
)

print("RECOVERY TEST: PASSED")


# ============================================================
# TEST 5 — STATUS
# ============================================================

print("\n" + "-" * 75)
print("TEST 5 — API STATUS")
print("-" * 75)

status = prime.get_status()

print(status)

assert status["step"] == 4

assert status["recovery_count"] == 1

print("STATUS TEST: PASSED")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("ALL STEP 53 TESTS PASSED")
print("=" * 75)

print(
    "\nPROJECT PRIME API STATUS:"
)

print("✓ PRIME core exposed through API")

print("✓ Real sensor sequence accepted")

print("✓ GNSS mode exposed")

print("✓ PRIME DR mode exposed")

print("✓ Recovery exposed")

print("✓ Position exposed")

print("✓ Status endpoint logic ready")

print("=" * 75)