import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from prime_core.gnss_monitor import GNSSMonitor
from prime_core.position_engine import PositionEngine
from prime_core.recovery_manager import RecoveryManager


print("=" * 70)
print("PROJECT PRIME — STEP 47")
print("RECOVERY STATE MACHINE TEST")
print("=" * 70)


# ============================================================
# CREATE COMPONENTS
# ============================================================

monitor = GNSSMonitor()

position = PositionEngine(
    52.4016600,
    -1.5052900
)

recovery = RecoveryManager()


# ============================================================
# INITIAL POSITION
# ============================================================

print("\nINITIAL POSITION")

print(
    position.get_position()
)


# ============================================================
# GNSS GOOD
# ============================================================

state = monitor.evaluate(
    satellites=10,
    accuracy=5.0,
    time_since_update=2.0
)

print(
    "\nGNSS state:",
    state
)


# ============================================================
# SIMULATE PRIME MOVEMENT
# ============================================================

print("\nSIMULATING PRIME DRIFT...")

position.update(
    100,
    -150
)

print(
    "PRIME position:",
    position.get_position()
)


# ============================================================
# GNSS OUTAGE
# ============================================================

state = monitor.evaluate(
    satellites=0,
    accuracy=None,
    time_since_update=20
)

print(
    "\nGNSS state:",
    state
)

print(
    "Mode: PRIME_DR"
)


# ============================================================
# MORE PRIME DRIFT
# ============================================================

position.update(
    200,
    -300
)

print(
    "\nPosition after outage:",
    position.get_position()
)


# ============================================================
# GNSS RETURNS
# ============================================================

recovered_lat = 52.4050000
recovered_lon = -1.5100000

state = monitor.evaluate(
    satellites=10,
    accuracy=5.0,
    time_since_update=2.0
)

print(
    "\nGNSS recovered:",
    state
)


# ============================================================
# RE-ANCHOR
# ============================================================

recovery.start_recovery()

success = recovery.reanchor(
    position,
    recovered_lat,
    recovered_lon
)

print(
    "Re-anchor successful:",
    success
)

print(
    "Recovery count:",
    recovery.recovery_count
)

print(
    "Final PRIME position:",
    position.get_position()
)


# ============================================================
# VERIFY
# ============================================================

final_position = (
    position.get_position()
)

lat_ok = abs(
    final_position["latitude"]
    -
    recovered_lat
) < 1e-9

lon_ok = abs(
    final_position["longitude"]
    -
    recovered_lon
) < 1e-9


print("\n" + "=" * 70)

if lat_ok and lon_ok:

    print(
        "RECOVERY TEST: PASSED"
    )

    print(
        "PRIME successfully re-anchored "
        "to recovered GNSS."
    )

else:

    print(
        "RECOVERY TEST: FAILED"
    )

print("=" * 70)
print("STEP 47 COMPLETE")
print("=" * 70)