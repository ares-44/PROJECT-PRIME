import numpy as np

from prime_core.gnss_monitor import GNSSMonitor
from prime_core.inference_engine import InferenceEngine
from prime_core.position_engine import PositionEngine
from prime_core.recovery_manager import RecoveryManager


class PRIMESystem:
    """
    PROJECT PRIME main navigation system.

    Pipeline:

    GNSS Monitor
          ↓
    Navigation Decision
       ↙       ↘
     GNSS     PRIME DR
                ↓
             CNN V2
                ↓
        Position Engine
                ↓
          GNSS Recovery
                ↓
             Re-anchor
    """

    def __init__(
        self,
        initial_latitude,
        initial_longitude
    ):

        # ----------------------------------------------------
        # CORE COMPONENTS
        # ----------------------------------------------------

        self.gnss_monitor = GNSSMonitor()

        self.inference_engine = (
            InferenceEngine()
        )

        self.position_engine = (
            PositionEngine(
                initial_latitude,
                initial_longitude
            )
        )

        self.recovery_manager = (
            RecoveryManager()
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.mode = "GNSS"

        self.previous_gnss_state = None

        self.step_count = 0

        self.recovery_count = 0

    # ========================================================
    # PROCESS SENSOR + GNSS INPUT
    # ========================================================

    def process(
        self,
        sensor_sequence,
        satellites,
        accuracy,
        time_since_update,
        gnss_latitude=None,
        gnss_longitude=None
    ):

        self.step_count += 1

        # ----------------------------------------------------
        # 1. GNSS HEALTH
        # ----------------------------------------------------

        gnss_state = (
            self.gnss_monitor.evaluate(
                satellites,
                accuracy,
                time_since_update
            )
        )

        recovered = False

        north = 0.0

        east = 0.0

        uncertainty = None

        # ====================================================
        # 2. GNSS GOOD
        # ====================================================

        if gnss_state == "GOOD":

            # -----------------------------------------------
            # Detect recovery from outage
            # -----------------------------------------------

            if (
                self.previous_gnss_state
                == "OUTAGE"
            ):

                self.mode = "RECOVERY"

                self.recovery_manager.start_recovery()

                success = (
                    self.recovery_manager.reanchor(
                        self.position_engine,
                        gnss_latitude,
                        gnss_longitude
                    )
                )

                if success:

                    recovered = True

                    self.recovery_count += 1

                    self.mode = "GNSS"

            else:

                # -------------------------------------------
                # Normal GNSS operation
                # -------------------------------------------

                self.mode = "GNSS"

                if (
                    gnss_latitude is not None
                    and
                    gnss_longitude is not None
                ):

                    self.position_engine.reset(
                        gnss_latitude,
                        gnss_longitude
                    )

        # ====================================================
        # 3. GNSS DEGRADED
        # ====================================================

        elif gnss_state == "DEGRADED":

            self.mode = "GNSS_DEGRADED"

        # ====================================================
        # 4. GNSS OUTAGE
        # ====================================================

        else:

            self.mode = "PRIME_DR"

            # -----------------------------------------------
            # CNN V2 inference
            # -----------------------------------------------

            prediction = (
                self.inference_engine.predict(
                    sensor_sequence
                )
            )

            north = float(
                prediction[0]
            )

            east = float(
                prediction[1]
            )

            # -----------------------------------------------
            # Update PRIME position
            # -----------------------------------------------

            self.position_engine.update(
                north,
                east
            )

            # -----------------------------------------------
            # Model uncertainty if available
            # -----------------------------------------------

            if hasattr(
                self.inference_engine,
                "last_uncertainty"
            ):

                uncertainty = (
                    self.inference_engine.last_uncertainty
                )

        # ====================================================
        # 5. CURRENT POSITION
        # ====================================================

        position = (
            self.position_engine.get_position()
        )

        # ====================================================
        # 6. SAVE STATE
        # ====================================================

        self.previous_gnss_state = (
            gnss_state
        )

        # ====================================================
        # 7. RETURN SYSTEM OUTPUT
        # ====================================================

        return {
            "step": self.step_count,

            "mode": self.mode,

            "gnss_state": gnss_state,

            "latitude":
                position["latitude"],

            "longitude":
                position["longitude"],

            "north_displacement":
                north,

            "east_displacement":
                east,

            "uncertainty":
                uncertainty,

            "recovered":
                recovered,

            "recovery_count":
                self.recovery_count
        }