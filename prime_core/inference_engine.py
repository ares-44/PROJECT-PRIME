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
            "prime_temporal_cnn_v2.keras"
        )

        scaler_path = os.path.join(
            base_dir,
            "models",
            "prime_sequence_scaler_v2.pkl"
        )

        print("Loading PRIME CNN V2...")

        self.model = keras.models.load_model(
            model_path
        )

        with open(
            scaler_path,
            "rb"
        ) as f:

            self.scaler = pickle.load(f)

        self.last_uncertainty = None

        print("PRIME CNN V2 loaded.")

    def predict(
        self,
        sensor_sequence
    ):

        sensor_sequence = np.asarray(
            sensor_sequence,
            dtype=np.float32
        )

        # Expected shape:
        # (60, 12)

        if sensor_sequence.shape != (60, 12):

            raise ValueError(
                f"Expected sensor sequence "
                f"shape (60, 12), got "
                f"{sensor_sequence.shape}"
            )

        # Scale each timestep independently
        original_shape = (
            sensor_sequence.shape
        )

        scaled = self.scaler.transform(
            sensor_sequence
        )

        scaled = scaled.reshape(
            original_shape
        )

        # CNN expects batch dimension
        prediction = self.model.predict(
            scaled[np.newaxis, :, :],
            verbose=0
        )[0]

        # Prediction:
        # [north displacement, east displacement]

        return prediction