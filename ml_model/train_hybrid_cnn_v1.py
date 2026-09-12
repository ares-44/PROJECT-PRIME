import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

print("=" * 75)
print("PROJECT PRIME — HYBRID CNN V1")
print("=" * 75)

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA = os.path.join(
    BASE,
    "data",
    "prime_hybrid_dataset_v1.npz"
)

MODEL_OUT = os.path.join(
    BASE,
    "models",
    "prime_hybrid_cnn_v1.keras"
)

SEQ_SCALER_OUT = os.path.join(
    BASE,
    "models",
    "prime_hybrid_sequence_scaler_v1.pkl"
)

PHYSICS_SCALER_OUT = os.path.join(
    BASE,
    "models",
    "prime_hybrid_physics_scaler_v1.pkl"
)

PRED_OUT = os.path.join(
    BASE,
    "data",
    "prime_hybrid_cnn_v1_test_predictions.csv"
)

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# LOAD
# ============================================================

data = np.load(
    DATA,
    allow_pickle=True
)

X_seq = data["X"]
X_phys = data["physics_features"]
y = data["y"]

print("\n===== DATA =====")
print("Sequence:", X_seq.shape)
print("Physics :", X_phys.shape)
print("Target  :", y.shape)

n = len(y)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

Xseq_train = X_seq[:train_end]
Xseq_val = X_seq[train_end:val_end]
Xseq_test = X_seq[val_end:]

Xphys_train = X_phys[:train_end]
Xphys_val = X_phys[train_end:val_end]
Xphys_test = X_phys[val_end:]

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

print("\n===== CHRONOLOGICAL SPLIT =====")
print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))

# ============================================================
# SCALE SEQUENCE
# ============================================================

seq_scaler = StandardScaler()

Xseq_train_flat = Xseq_train.reshape(
    -1,
    Xseq_train.shape[-1]
)

Xseq_val_flat = Xseq_val.reshape(
    -1,
    Xseq_val.shape[-1]
)

Xseq_test_flat = Xseq_test.reshape(
    -1,
    Xseq_test.shape[-1]
)

seq_scaler.fit(
    Xseq_train_flat
)

Xseq_train = seq_scaler.transform(
    Xseq_train_flat
).reshape(
    Xseq_train.shape
)

Xseq_val = seq_scaler.transform(
    Xseq_val_flat
).reshape(
    Xseq_val.shape
)

Xseq_test = seq_scaler.transform(
    Xseq_test_flat
).reshape(
    Xseq_test.shape
)

# ============================================================
# SCALE PHYSICS FEATURES
# ============================================================

phys_scaler = StandardScaler()

phys_scaler.fit(
    Xphys_train
)

Xphys_train = phys_scaler.transform(
    Xphys_train
)

Xphys_val = phys_scaler.transform(
    Xphys_val
)

Xphys_test = phys_scaler.transform(
    Xphys_test
)

# ============================================================
# MODEL
# ============================================================

print("\n===== MODEL =====")

sequence_input = Input(
    shape=(
        Xseq_train.shape[1],
        Xseq_train.shape[2]
    ),
    name="sensor_sequence"
)

x = Conv1D(
    32,
    5,
    padding="same",
    activation="relu"
)(sequence_input)

x = BatchNormalization()(x)

x = Conv1D(
    64,
    5,
    padding="same",
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Conv1D(
    64,
    3,
    padding="same",
    activation="relu"
)(x)

x = GlobalAveragePooling1D()(x)

x = Dense(
    64,
    activation="relu"
)(x)

physics_input = Input(
    shape=(
        Xphys_train.shape[1],
    ),
    name="physics_features"
)

p = Dense(
    32,
    activation="relu"
)(physics_input)

p = Dense(
    16,
    activation="relu"
)(p)

# ============================================================
# FUSION
# ============================================================

combined = Concatenate()(
    [x, p]
)

combined = Dense(
    64,
    activation="relu"
)(combined)

combined = Dropout(
    0.25
)(combined)

combined = Dense(
    32,
    activation="relu"
)(combined)

output = Dense(
    2,
    name="displacement"
)(combined)

model = Model(
    inputs=[
        sequence_input,
        physics_input
    ],
    outputs=output
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

model.fit(
    [
        Xseq_train,
        Xphys_train
    ],
    y_train,

    validation_data=(
        [
            Xseq_val,
            Xphys_val
        ],
        y_val
    ),

    epochs=150,

    batch_size=32,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# PREDICT
# ============================================================

print("\n===== PREDICTIONS =====")

val_pred = model.predict(
    [
        Xseq_val,
        Xphys_val
    ],
    verbose=0
)

test_pred = model.predict(
    [
        Xseq_test,
        Xphys_test
    ],
    verbose=0
)

# ============================================================
# EVALUATION
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

    distance_error = np.abs(
        np.linalg.norm(
            predicted,
            axis=1
        )
        -
        np.linalg.norm(
            actual,
            axis=1
        )
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

    return error_2d


val_errors = evaluate(
    "VALIDATION",
    y_val,
    val_pred
)

test_errors = evaluate(
    "TEST",
    y_test,
    test_pred
)

# ============================================================
# BENCHMARK
# ============================================================

print(
    "\n===== CURRENT BEST BENCHMARK ====="
)

print(
    "CNN V2 Mean   : 43.8872 m"
)

print(
    "CNN V2 Median : 40.5415 m"
)

print(
    "CNN V2 P90    : 75.4883 m"
)

print(
    "CNN V2 Max    : 113.4356 m"
)

mean_change = (
    (
        43.8872
        -
        np.mean(test_errors)
    )
    /
    43.8872
) * 100

p90_change = (
    (
        75.4883
        -
        np.percentile(
            test_errors,
            90
        )
    )
    /
    75.4883
) * 100

print(
    "\n===== COMPARISON ====="
)

print(
    "Mean error change:",
    round(
        mean_change,
        2
    ),
    "%"
)

print(
    "P90 error change:",
    round(
        p90_change,
        2
    ),
    "%"
)

# ============================================================
# SAVE
# ============================================================

model.save(
    MODEL_OUT
)

with open(
    SEQ_SCALER_OUT,
    "wb"
) as f:

    pickle.dump(
        seq_scaler,
        f
    )

with open(
    PHYSICS_SCALER_OUT,
    "wb"
) as f:

    pickle.dump(
        phys_scaler,
        f
    )

pd.DataFrame(
    {
        "actual_north_m":
            y_test[:, 0],

        "actual_east_m":
            y_test[:, 1],

        "predicted_north_m":
            test_pred[:, 0],

        "predicted_east_m":
            test_pred[:, 1],

        "error_2d_m":
            test_errors
    }
).to_csv(
    PRED_OUT,
    index=False
)

print(
    "\n===== SAVED ====="
)

print(
    "Model:",
    MODEL_OUT
)

print(
    "Sequence scaler:",
    SEQ_SCALER_OUT
)

print(
    "Physics scaler:",
    PHYSICS_SCALER_OUT
)

print(
    "Predictions:",
    PRED_OUT
)

print(
    "\n" + "=" * 75
)

print(
    "STEP 34B COMPLETE"
)

print(
    "=" * 75
)