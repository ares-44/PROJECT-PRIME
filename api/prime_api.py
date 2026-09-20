import os
import sys
from typing import Optional

# Allow imports from PROJECT_PRIME
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from prime_core.prime_system import PRIMESystem


class PRIMEAPI:
    """
    Clean interface between PROJECT PRIME core
    and an external application/frontend.

    Frontend does not need to know about:
    - TensorFlow
    - CNN architecture
    - scalers
    - GNSS thresholds
    - position calculations
    """

    def __init__(
        self,
        initial_latitude: float,
        initial_longitude: float
    ):

        self.system = PRIMESystem(
            initial_latitude,
            initial_longitude
        )

        # Track whether the real phone GNSS position
        # has been used to establish the PRIME anchor.
        self.gnss_initialized = False

    def process_navigation(
        self,
        sensor_sequence,
        satellites: Optional[int] = None,
        accuracy: Optional[float] = None,
        time_since_update: float = 0.0,
        gnss_latitude: Optional[float] = None,
        gnss_longitude: Optional[float] = None,
        step_dt: float = 1.0
    ):
        """
        Process one navigation update.

        Parameters
        ----------
        sensor_sequence:
            60 x 25 PRIME V7 sensor sequence.

        satellites:
            Number of visible GNSS satellites.

        accuracy:
            GNSS accuracy in meters.

        time_since_update:
            Seconds since the last GNSS position update.

        gnss_latitude:
            Current GNSS latitude if available.

        gnss_longitude:
            Current GNSS longitude if available.

        step_dt:
            Elapsed time of this step in seconds.
        """

        # ====================================================
        # INITIAL PHONE GNSS ANCHOR
        # ====================================================
        #
        # If the first request already represents GNSS outage,
        # use the phone's last known GPS position as the PRIME
        # starting point before running dead reckoning.
        #
        if (
            not self.gnss_initialized
            and gnss_latitude is not None
            and gnss_longitude is not None
        ):

            self.system.position_engine.reset(
                gnss_latitude,
                gnss_longitude
            )

            self.gnss_initialized = True

            print(
                "PRIME initial GNSS anchor: "
                f"{gnss_latitude}, {gnss_longitude}"
            )

        # ====================================================
        # PROCESS NAVIGATION
        # ====================================================

        result = self.system.process(
            sensor_sequence=sensor_sequence,
            satellites=satellites,
            accuracy=accuracy,
            time_since_update=time_since_update,
            gnss_latitude=gnss_latitude,
            gnss_longitude=gnss_longitude,
            step_dt=step_dt
        )

        # If a valid GNSS fix is received later,
        # consider the system initialized.
        if (
            gnss_latitude is not None
            and gnss_longitude is not None
            and satellites is not None
            and satellites > 0
        ):
            self.gnss_initialized = True

        return self._format_response(
            result
        )

    def get_position(self):

        return (
            self.system
            .position_engine
            .get_position()
        )

    def get_status(self):

        return {
            "mode":
                self.system.mode,

            "gnss_state":
                self.system.previous_gnss_state,

            "recovery_count":
                self.system.recovery_count,

            "step":
                self.system.step_count,

            "gnss_initialized":
                self.gnss_initialized
        }

    def reset(
        self,
        latitude: float,
        longitude: float
    ):

        self.system.position_engine.reset(
            latitude,
            longitude
        )

        self.system.mode = "GNSS"

        self.system.previous_gnss_state = "GOOD"

        self.gnss_initialized = True

        return {
            "success": True,
            "latitude": latitude,
            "longitude": longitude
        }

    @staticmethod
    def _format_response(result):

        return {
            "success": True,

            "navigation": {
                "mode":
                    result["mode"],

                "gnss_state":
                    result["gnss_state"],

                "latitude":
                    result["latitude"],

                "longitude":
                    result["longitude"]
            },

            "movement": {
                "north_displacement_m":
                    result[
                        "north_displacement"
                    ],

                "east_displacement_m":
                    result[
                        "east_displacement"
                    ]
            },

            "recovery": {
                "recovered":
                    result["recovered"],

                "recovery_count":
                    result[
                        "recovery_count"
                    ]
            },

            "uncertainty":
                result["uncertainty"],

            "step":
                result["step"]
        }