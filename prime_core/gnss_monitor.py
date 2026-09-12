class GNSSMonitor:

    def __init__(
        self,
        degraded_accuracy=20.0,
        outage_timeout=15.0,
        degraded_timeout=10.0,
        min_satellites=4
    ):

        self.degraded_accuracy = (
            degraded_accuracy
        )

        self.outage_timeout = (
            outage_timeout
        )

        self.degraded_timeout = (
            degraded_timeout
        )

        self.min_satellites = (
            min_satellites
        )

    def evaluate(
        self,
        satellites,
        accuracy,
        time_since_update
    ):

        # Complete outage
        if (
            satellites is None
            or accuracy is None
            or time_since_update >= self.outage_timeout
        ):

            return "OUTAGE"

        # Degraded GNSS
        if (
            satellites < self.min_satellites
            or accuracy > self.degraded_accuracy
            or time_since_update >= self.degraded_timeout
        ):

            return "DEGRADED"

        return "GOOD"