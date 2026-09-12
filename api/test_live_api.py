import numpy as np
import requests

# Load real PRIME sensor dataset
data = np.load("data/prime_sequence_dataset_v2.npz")

X = data["X"]

print("Dataset shape:", X.shape)

# Take first real 60x12 sensor sequence
sequence = X[0].tolist()

payload = {
    "sensor_data": {
        "sequence": sequence
    },
    "satellites": 8,
    "accuracy": 5.0,
    "time_since_update": 20.0,
    "gnss_latitude": None,
    "gnss_longitude": None
}

print("\nSending sequence to PRIME API...")

response = requests.post(
    "http://127.0.0.1:8000/navigate",
    json=payload
)

print("\nHTTP Status:", response.status_code)
print("Response:")
print(response.json())