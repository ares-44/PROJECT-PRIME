from prime_core.recovery_manager import RecoveryManager


class PRIMENavigation:

    GNSS_MODE = "GNSS"
    PRIME_MODE = "PRIME_DR"
    DEGRADED_MODE = "GNSS_DEGRADED"
    RECOVERY_MODE = "RECOVERY"

    def __init__(
        self,
        gnss_monitor,
        inference_engine,
        position_engine
    ):

        self.gnss_monitor = gnss_monitor
        self.inference_engine = inference_engine
        self.position_engine = position_engine

        self.recovery_manager = (
            RecoveryManager()
        )

        self.mode = self.GNSS_MODE
        self.previous_gnss_state = None

    def update(
        self,
        sensor_sequence,
        satellites,
        accuracy,
        time_since_update,
        gnss_latitude=None,
        gnss_longitude=None
    ):

        gnss_state = self.gnss_monitor.evaluate(
            satellites,
            accuracy,
            time_since_update
        )

        # ====================================================
        # GNSS GOOD
        # ====================================================

        if gnss_state == "GOOD":

            # Detect return from outage
            if self.previous_gnss_state == "OUTAGE":

                self.mode = self.RECOVERY_MODE

                self.recovery_manager.start_recovery()

                success = (
                    self.recovery_manager.reanchor(
                        self.position_engine,
                        gnss_latitude,
                        gnss_longitude
                    )
                )

                if success:

                    self.mode = self.GNSS_MODE

            else:

                self.mode = self.GNSS_MODE

            self.previous_gnss_state = gnss_state

            return {
                "mode": self.mode,
                "gnss_state": gnss_state,
                "position":
                    self.position_engine.get_position(),
                "recovery":
                    self.recovery_manager.recovery_count
            }

        # ====================================================
        # GNSS DEGRADED
        # ====================================================

        if gnss_state == "DEGRADED":

            self.mode = self.DEGRADED_MODE

            self.previous_gnss_state = gnss_state

            return {
                "mode": self.mode,
                "gnss_state": gnss_state,
                "position":
                    self.position_engine.get_position(),
                "recovery":
                    self.recovery_manager.recovery_count
            }

        # ====================================================
        # GNSS OUTAGE
        # ====================================================

        self.mode = self.PRIME_MODE

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

        position = (
            self.position_engine.update(
                north,
                east
            )
        )

        self.previous_gnss_state = "OUTAGE"

        return {
            "mode": self.mode,
            "gnss_state": gnss_state,
            "north_displacement": north,
            "east_displacement": east,
            "position": position,
            "recovery":
                self.recovery_manager.recovery_count
        }