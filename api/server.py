import os
import sys
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from api.prime_api import PRIMEAPI


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="PROJECT PRIME API",
    description=(
        "AI-powered GNSS-denied navigation "
        "and dead-reckoning API using PRIME Temporal CNN V7."
    ),
    version="1.0.0"
)


# ============================================================
# GLOBAL PRIME SYSTEM
# ============================================================

prime = PRIMEAPI(
    initial_latitude=52.4016600,
    initial_longitude=-1.5052900
)


# ============================================================
# REQUEST MODELS
# ============================================================

class SensorData(BaseModel):

    sequence: list[list[float]] = Field(
        ...,
        description="60 x 25 PRIME V7 feature sequence"
    )


class NavigationRequest(BaseModel):

    sensor_data: SensorData

    satellites: int | None = None

    accuracy: float | None = None

    time_since_update: float = 0.0

    gnss_latitude: float | None = None

    gnss_longitude: float | None = None

    step_dt: float = 1.0


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "PROJECT PRIME",
        "status": "online",
        "version": "1.0.0",
        "model": "PRIME Temporal CNN V7"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "project": "PROJECT PRIME",
        "model": "PRIME Temporal CNN V7"
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/status")
def status():

    return prime.get_status()


# ============================================================
# CURRENT POSITION
# ============================================================

@app.get("/position")
def position():

    return prime.get_position()


# ============================================================
# NAVIGATION
# ============================================================

@app.post("/navigate")
def navigate(
    request: NavigationRequest
):

    sequence = np.asarray(
        request.sensor_data.sequence,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Validate V7 shape
    # --------------------------------------------------------

    if sequence.shape != (60, 25):

        raise HTTPException(
            status_code=400,
            detail=(
                "Sensor sequence must have "
                "shape (60, 25). "
                f"Received {sequence.shape}."
            )
        )

    # --------------------------------------------------------
    # PRIME V7 inference
    # --------------------------------------------------------

    try:

        result = prime.process_navigation(

            sensor_sequence=sequence,

            satellites=request.satellites,

            accuracy=request.accuracy,

            time_since_update=
                request.time_since_update,

            gnss_latitude=
                request.gnss_latitude,

            gnss_longitude=
                request.gnss_longitude,

            step_dt=
                request.step_dt
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RESET / RE-ANCHOR
# ============================================================

@app.post("/reset")
def reset(
    latitude: float,
    longitude: float
):

    return prime.reset(
        latitude,
        longitude
    )