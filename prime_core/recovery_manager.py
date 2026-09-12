class RecoveryManager:
    """
    Handles GNSS recovery and re-anchoring.

    When reliable GNSS returns, PRIME's propagated
    position is replaced by the recovered GNSS position.
    """

    RECOVERY = "RECOVERY"

    def __init__(self):
        self.active = False
        self.recovery_count = 0

    def start_recovery(self):
        self.active = True

    def reanchor(
        self,
        position_engine,
        gnss_latitude,
        gnss_longitude
    ):
        """
        Re-anchor PRIME to the recovered GNSS position.
        """

        if gnss_latitude is None or gnss_longitude is None:
            return False

        position_engine.reset(
            gnss_latitude,
            gnss_longitude
        )

        self.active = False
        self.recovery_count += 1

        return True

    def is_active(self):
        return self.active