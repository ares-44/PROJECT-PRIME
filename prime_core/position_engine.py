import math


class PositionEngine:
    """
    Converts predicted North/East displacement
    into a new latitude/longitude position.
    """

    EARTH_RADIUS_M = 6371000.0

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def update(self, north_m, east_m):
        """
        Update position using displacement in meters.

        north_m: movement toward North (+) / South (-)
        east_m: movement toward East (+) / West (-)
        """

        lat_rad = math.radians(self.latitude)

        delta_lat = (
            north_m / self.EARTH_RADIUS_M
        )

        delta_lon = (
            east_m /
            (self.EARTH_RADIUS_M * math.cos(lat_rad))
        )

        self.latitude += math.degrees(delta_lat)
        self.longitude += math.degrees(delta_lon)

        return self.get_position()

    def get_position(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude
        }

    def reset(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude