import os
import sys
import numpy as np


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from prime_core.prime_system import PRIMESystem


print("=" * 75)
print("PROJECT PRIME — STEP 50")
print("UNIFIED PRIME SYSTEM TEST")
print("=" * 75)


# ============================================================
# INITIAL POSITION
# ============================================================

initial_lat = 52.4016600
initial_lon = -1.5052900


prime = PRIMESystem(
    initial_lat,
    initial_lon
)


print("\nPRIME SYSTEM INITIALIZED")

print(
    prime.position_engine.get_position()
)


# ============================================================
# DUMMY SENSOR SEQUENCE
# ============================================================

# 60 timesteps × 12 sensor channels

sensor_sequence = np.zeros(
    (60, 12),
    dtype=np.float32
)


# ============================================================
# TEST 1 — GNSS GOOD
# ============================================================

print("\n" + "-" * 75)

print("TEST 1 — GNSS GOOD")

print("-" * 75)


result = prime.process(
    sensor_sequence=sensor_sequence,

    satellites=10,

    accuracy=5.0,

    time_since_update=2.0,

    gnss_latitude=52.4016600,

    gnss_longitude=-1.5052900
)


print(result)


assert result["mode"] == "GNSS"

assert result["gnss_state"] == "GOOD"


print("GNSS TEST: PASSED")


# ============================================================
# TEST 2 — GNSS DEGRADED
# ============================================================

print("\n" + "-" * 75)

print("TEST 2 — GNSS DEGRADED")

print("-" * 75)


result = prime.process(
    sensor_sequence=sensor_sequence,

    satellites=3,

    accuracy=8.0,

    time_since_update=5.0
)


print(result)


assert result["mode"] == "GNSS_DEGRADED"

assert result["gnss_state"] == "DEGRADED"


print("DEGRADED TEST: PASSED")


# ============================================================
# TEST 3 — GNSS OUTAGE
# ============================================================

print("\n" + "-" * 75)

print("TEST 3 — GNSS OUTAGE")

print("-" * 75)


result = prime.process(
    sensor_sequence=sensor_sequence,

    satellites=0,

    accuracy=None,

    time_since_update=20.0
)


print(result)


assert result["mode"] == "PRIME_DR"

assert result["gnss_state"] == "OUTAGE"


print("OUTAGE TEST: PASSED")


# ============================================================
# TEST 4 — SECOND OUTAGE STEP
# ============================================================

print("\n" + "-" * 75)

print("TEST 4 — CONTINUOUS PRIME DR")

print("-" * 75)


result = prime.process(
    sensor_sequence=sensor_sequence,

    satellites=0,

    accuracy=None,

    time_since_update=20.0
)


print(result)


assert result["mode"] == "PRIME_DR"

assert result["recovered"] is False


print("PRIME DR TEST: PASSED")


# ============================================================
# TEST 5 — GNSS RECOVERY
# ============================================================

print("\n" + "-" * 75)

print("TEST 5 — GNSS RECOVERY + RE-ANCHOR")

print("-" * 75)


recovered_lat = 52.4050000

recovered_lon = -1.5100000


result = prime.process(
    sensor_sequence=sensor_sequence,

    satellites=10,

    accuracy=5.0,

    time_since_update=2.0,

    gnss_latitude=recovered_lat,

    gnss_longitude=recovered_lon
)


print(result)


assert result["mode"] == "GNSS"

assert result["gnss_state"] == "GOOD"

assert result["recovered"] is True

assert result["recovery_count"] == 1

assert abs(
    result["latitude"]
    -
    recovered_lat
) < 1e-9

assert abs(
    result["longitude"]
    -
    recovered_lon
) < 1e-9


print("RECOVERY TEST: PASSED")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)

print("ALL STEP 50 TESTS PASSED")

print("=" * 75)

print("\nPROJECT PRIME SYSTEM STATUS:")

print("✓ GNSS monitoring")

print("✓ GNSS degraded detection")

print("✓ GNSS outage detection")

print("✓ CNN-based PRIME DR")

print("✓ Position propagation")

print("✓ GNSS recovery")

print("✓ Automatic re-anchoring")

print("✓ Unified system interface")

print("=" * 75)