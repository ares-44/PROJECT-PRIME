from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import csv
from threading import Lock

app = FastAPI(title="PROJECT PRIME DATA COLLECTOR")

BASE = Path(__file__).resolve().parent
OUT = BASE / "phone_training_data.csv"
LOCK = Lock()

FIELDS = [
    "timestamp", "latitude", "longitude", "accuracy",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "rx", "ry", "rz",
    "mx", "my", "mz",
]


class Sample(BaseModel):
    timestamp: float
    latitude: float
    longitude: float
    accuracy: float | None = None
    sensor: list[float]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "collector": "PROJECT PRIME"
    }


@app.get("/count")
def count():
    if not OUT.exists():
        return {
            "samples": 0,
            "file": str(OUT)
        }

    with OUT.open("r", encoding="utf-8") as f:
        n = max(0, sum(1 for _ in f) - 1)

    return {
        "samples": n,
        "file": str(OUT)
    }


@app.post("/collect")
def collect(sample: Sample):

    if len(sample.sensor) != 12:
        return {
            "ok": False,
            "error": f"expected 12 sensor values, got {len(sample.sensor)}"
        }

    row = [
        sample.timestamp,
        sample.latitude,
        sample.longitude,
        sample.accuracy,
        *sample.sensor
    ]

    with LOCK:
        new_file = not OUT.exists()

        with OUT.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            if new_file:
                writer.writerow(FIELDS)

            writer.writerow(row)

    return {
        "ok": True
    }


@app.post("/clear")
def clear():

    with LOCK:
        if OUT.exists():
            OUT.unlink()

    return {
        "ok": True,
        "message": "collector CSV cleared"
    }
