import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


print("=" * 75)
print("PROJECT PRIME — TEMPORAL CNN V2")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA = os.path.join(
    BASE,
    "data",
    "prime_sequence_dataset_v2.npz"
)

MODEL_OUT = os.path.join(
    BASE,
    "models",
    "prime_temporal_cnn_v2.keras"
)

SCALER_OUT = os.path.join(
    BASE,
    "models",
    "prime_sequence_scaler_v2.pkl"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(DATA)

X = data["X"]
y = data["y"]

print("\n===== DATA =====")

print("X shape:", X.shape)
print("y shape:", y.shape)

n = len(X)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[
    train_end:val_end
]

y_val = y[
    train_end:val_end
]

X_test = X[
    val_end:
]

y_test = y[
    val_end:
]


print("\n===== CHRONOLOGICAL SPLIT =====")

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ============================================================
# SCALE SENSOR FEATURES
# ============================================================

# Fit scaler ONLY on training data.

scaler = StandardScaler()

train_shape = X_train.shape

X_train_flat = X_train.reshape(
    -1,
    X_train.shape[-1]
)

X_val_flat = X_val.reshape(
    -1,
    X_val.shape[-1]
)

X_test_flat = X_test.reshape(
    -1,
    X_test.shape[-1]
)


scaler.fit(
    X_train_flat
)


X_train_scaled = scaler.transform(
    X_train_flat
).reshape(
    X_train.shape
)

X_val_scaled = scaler.transform(
    X_val_flat
).reshape(
    X_val.shape
)

X_test_scaled = scaler.transform(
    X_test_flat
).reshape(
    X_test.shape
)


# ============================================================
# MODEL
# ============================================================

print("\n===== MODEL =====")

model = Sequential(
    [
        Input(
            shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        ),

        Conv1D(
            filters=32,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        Conv1D(
            filters=64,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        GlobalAveragePooling1D(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(
            0.25
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            2
        )
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n===== TRAINING =====")

history = model.fit(
    X_train_scaled,
    y_train,

    validation_data=(
        X_val_scaled,
        y_val
    ),

    epochs=150,

    batch_size=32,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n===== PREDICTIONS =====")

val_pred = model.predict(
    X_val_scaled,
    verbose=0
)

test_pred = model.predict(
    X_test_scaled,
    verbose=0
)


# ============================================================
# METRICS FUNCTION
# ============================================================

def evaluate(
    name,
    actual,
    predicted
):

    north_mae = mean_absolute_error(
        actual[:, 0],
        predicted[:, 0]
    )

    east_mae = mean_absolute_error(
        actual[:, 1],
        predicted[:, 1]
    )

    north_rmse = np.sqrt(
        mean_squared_error(
            actual[:, 0],
            predicted[:, 0]
        )
    )

    east_rmse = np.sqrt(
        mean_squared_error(
            actual[:, 1],
            predicted[:, 1]
        )
    )

    north_r2 = r2_score(
        actual[:, 0],
        predicted[:, 0]
    )

    east_r2 = r2_score(
        actual[:, 1],
        predicted[:, 1]
    )

    error_2d = np.linalg.norm(
        predicted - actual,
        axis=1
    )

    predicted_distance = np.linalg.norm(
        predicted,
        axis=1
    )

    actual_distance = np.linalg.norm(
        actual,
        axis=1
    )

    distance_error = np.abs(
        predicted_distance
        -
        actual_distance
    )

    print(
        f"\n===== {name} ====="
    )

    print(
        "North MAE :",
        round(north_mae, 4),
        "m"
    )

    print(
        "East MAE  :",
        round(east_mae, 4),
        "m"
    )

    print(
        "North RMSE:",
        round(north_rmse, 4),
        "m"
    )

    print(
        "East RMSE :",
        round(east_rmse, 4),
        "m"
    )

    print(
        "North R²  :",
        round(north_r2, 4)
    )

    print(
        "East R²   :",
        round(east_r2, 4)
    )

    print(
        "2D Mean Error:",
        round(
            np.mean(error_2d),
            4
        ),
        "m"
    )

    print(
        "2D Median Error:",
        round(
            np.median(error_2d),
            4
        ),
        "m"
    )

    print(
        "2D P90 Error:",
        round(
            np.percentile(
                error_2d,
                90
            ),
            4
        ),
        "m"
    )

    print(
        "2D Maximum Error:",
        round(
            np.max(error_2d),
            4
        ),
        "m"
    )

    print(
        "Distance MAE:",
        round(
            np.mean(distance_error),
            4
        ),
        "m"
    )

    return {
        "2d_mean": np.mean(error_2d),
        "2d_median": np.median(error_2d),
        "2d_p90": np.percentile(
            error_2d,
            90
        ),
        "2d_max": np.max(error_2d)
    }


# ============================================================
# VALIDATION
# ============================================================

val_results = evaluate(
    "VALIDATION",
    y_val,
    val_pred
)


# ============================================================
# TEST
# ============================================================

test_results = evaluate(
    "TEST",
    y_test,
    test_pred
)


# ============================================================
# COMPARE WITH RF V2
# ============================================================

print(
    "\n===== RF V2 BENCHMARK ====="
)

print(
    "RF V2 Test Mean   : 45.6124 m"
)

print(
    "RF V2 Test Median : 42.7093 m"
)

print(
    "RF V2 Test P90    : 78.3821 m"
)

print(
    "RF V2 Test Max    : 117.6400 m"
)


mean_improvement = (
    (
        45.6124
        -
        test_results["2d_mean"]
    )
    /
    45.6124
) * 100


p90_improvement = (
    (
        78.3821
        -
        test_results["2d_p90"]
    )
    /
    78.3821
) * 100


print(
    "\n===== RF COMPARISON ====="
)

print(
    "Mean error change:",
    round(
        mean_improvement,
        2
    ),
    "%"
)

print(
    "P90 error change:",
    round(
        p90_improvement,
        2
    ),
    "%"
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_OUT
)


with open(
    SCALER_OUT,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_out = os.path.join(
    BASE,
    "data",
    "prime_temporal_cnn_v2_test_predictions.csv"
)


prediction_df = np.column_stack(
    [
        y_test[:, 0],
        y_test[:, 1],
        test_pred[:, 0],
        test_pred[:, 1]
    ]
)


pd.DataFrame(
    prediction_df,
    columns=[
        "actual_north",
        "actual_east",
        "predicted_north",
        "predicted_east"
    ]
).to_csv(
    prediction_out,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n===== SAVED ====="
)

print(
    "Model:",
    MODEL_OUT
)

print(
    "Scaler:",
    SCALER_OUT
)

print(
    "Predictions:",
    prediction_out
)

print(
    "\n" + "=" * 75
)

print(
    "STEP 32B COMPLETE"
)

print(
    "=" * 75
)