import os
import pickle
import numpy as np
from tensorflow import keras


class InferenceEngine:

    def __init__(self):

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        model_path = os.path.join(
            base_dir,
            "models",
            "prime_temporal_cnn_v7.keras"
        )

        scaler_path = os.path.join(
            base_dir,
            "models",
            "prime_sequence_scaler_v7.pkl"
        )

        print("Loading PRIME CNN V7...")

        self.model = keras.models.load_model(
            model_path
        )

        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)

        self.sensor_scaler = scalers["sensor_scaler"]
        self.physics_scaler = scalers["physics_scaler"]

        self.last_uncertainty = None

        print("PRIME CNN V7 loaded.")

    def predict(
        self,
        sensor_sequence,
        physics_input=None
    ):

        sensor_sequence = np.asarray(
            sensor_sequence,
            dtype=np.float32
        )

        if sensor_sequence.shape != (60, 25):
            raise ValueError(
                f"Expected sensor sequence shape (60, 25), "
                f"got {sensor_sequence.shape}"
            )

        # ----------------------------------------------------
        # PHYSICS INPUT
        # ----------------------------------------------------

        if physics_input is None:

            # Use the final physics displacement
            # contained in the V7 feature sequence (channels 20 & 21).
            physics_input = sensor_sequence[-1, 20:22]

        physics_input = np.asarray(
            physics_input,
            dtype=np.float32
        ).reshape(1, 2)

        # ----------------------------------------------------
        # SCALE SENSOR DATA
        # ----------------------------------------------------

        scaled_sensor = self.sensor_scaler.transform(
            sensor_sequence
        ).reshape(1, 60, 25)

        # ----------------------------------------------------
        # SCALE PHYSICS INPUT
        # ----------------------------------------------------

        scaled_physics = self.physics_scaler.transform(
            physics_input
        )

        # ----------------------------------------------------
        # V7 INFERENCE
        # ----------------------------------------------------

        prediction = self.model.predict(
            [
                scaled_sensor,
                scaled_physics
            ],
            verbose=0
        )[0]

        north = float(prediction[0])
        east = float(prediction[1])

        print(
            f"PRIME V7 prediction: "
            f"N={north:.3f} m, "
            f"E={east:.3f} m"
        )

        return np.array(
            [north, east],
            dtype=np.float32
        )