import numpy as np
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Load real PRIME sensor data
data = np.load("data/prime_sequence_dataset_v2.npz")
X = data["X"]

# Use one real 60x12 sensor sequence
sequence = X[0].tolist()


def send_request(name, satellites, accuracy, time_since_update,
                 latitude=None, longitude=None):

    payload = {
        "sensor_data": {
            "sequence": sequence
        },
        "satellites": satellites,
        "accuracy": accuracy,
        "time_since_update": time_since_update,
        "gnss_latitude": latitude,
        "gnss_longitude": longitude
    }

    response = requests.post(
        f"{BASE_URL}/navigate",
        json=payload
    )

    print(f"\n===== {name} =====")
    print("HTTP:", response.status_code)

    result = response.json()
    print(json.dumps(result, indent=2))

    return result


print("PROJECT PRIME - GNSS RECOVERY TEST")
print("Dataset:", X.shape)


# --------------------------------------------------
# STEP 1: GNSS OUTAGE
# --------------------------------------------------

send_request(
    "STEP 1 - GNSS OUTAGE",
    satellites=0,
    accuracy=999,
    time_since_update=20
)


# --------------------------------------------------
# STEP 2: CONTINUE OUTAGE
# --------------------------------------------------

send_request(
    "STEP 2 - CONTINUOUS PRIME DR",
    satellites=0,
    accuracy=999,
    time_since_update=25
)


# --------------------------------------------------
# STEP 3: CONTINUE OUTAGE
# --------------------------------------------------

send_request(
    "STEP 3 - CONTINUOUS PRIME DR",
    satellites=0,
    accuracy=999,
    time_since_update=30
)


# --------------------------------------------------
# STEP 4: GNSS RECOVERY
# --------------------------------------------------

send_request(
    "STEP 4 - GNSS RECOVERY",
    satellites=8,
    accuracy=5.0,
    time_since_update=1,
    latitude=52.405000,
    longitude=-1.510000
)


# --------------------------------------------------
# FINAL STATUS
# --------------------------------------------------

response = requests.get(f"{BASE_URL}/status")

print("\n===== FINAL SYSTEM STATUS =====")
print(json.dumps(response.json(), indent=2))