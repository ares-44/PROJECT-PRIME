import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Input
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("PROJECT PRIME — TEMPORAL CNN V1")
print("=" * 70)


# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_sequence_dataset_v1.npz"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_temporal_cnn_v1.keras"

SCALER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_sequence_scaler_v1.pkl"


# =========================================================
# LOAD DATA
# =========================================================

data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)

y = data["y"].astype(np.float32)

print("\n===== DATA =====")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================================================
# CHRONOLOGICAL SPLIT
# =========================================================

n = len(X)

train_end = int(n * 0.70)

val_end = int(n * 0.85)


X_train = X[:train_end]

y_train = y[:train_end]


X_val = X[train_end:val_end]

y_val = y[train_end:val_end]


X_test = X[val_end:]

y_test = y[val_end:]


print("\n===== SPLIT =====")

print(
    "Train:",
    len(X_train)
)

print(
    "Validation:",
    len(X_val)
)

print(
    "Test:",
    len(X_test)
)


# =========================================================
# SENSOR NORMALIZATION
# =========================================================

# Fit scaler ONLY on training data

scaler = StandardScaler()

train_flat = X_train.reshape(
    -1,
    X_train.shape[-1]
)

scaler.fit(train_flat)


def scale_sequences(X):

    shape = X.shape

    flat = X.reshape(
        -1,
        shape[-1]
    )

    scaled = scaler.transform(
        flat
    )

    return scaled.reshape(
        shape
    ).astype(np.float32)


X_train = scale_sequences(
    X_train
)

X_val = scale_sequences(
    X_val
)

X_test = scale_sequences(
    X_test
)


# =========================================================
# MODEL
# =========================================================

model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    GlobalAveragePooling1D(),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    Dense(
        2
    )
])


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


print("\n===== MODEL =====")

model.summary()


# =========================================================
# TRAIN
# =========================================================

print("\n===== TRAINING =====")


early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=20,

    restore_best_weights=True
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=200,

    batch_size=16,

    callbacks=[
        early_stop
    ],

    verbose=1
)


# =========================================================
# VALIDATION
# =========================================================

val_pred = model.predict(
    X_val,
    verbose=0
)


print("\n===== VALIDATION =====")


for i, name in enumerate(
    ["north", "east"]
):

    mae = mean_absolute_error(
        y_val[:, i],
        val_pred[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val[:, i],
            val_pred[:, i]
        )
    )

    r2 = r2_score(
        y_val[:, i],
        val_pred[:, i]
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        "MAE:",
        round(mae, 4),
        "m"
    )

    print(
        "RMSE:",
        round(rmse, 4),
        "m"
    )

    print(
        "R2:",
        round(r2, 4)
    )


# =========================================================
# FINAL TEST
# =========================================================

test_pred = model.predict(
    X_test,
    verbose=0
)


print("\n===== FINAL TEST =====")


for i, name in enumerate(
    ["north", "east"]
):

    mae = mean_absolute_error(
        y_test[:, i],
        test_pred[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            test_pred[:, i]
        )
    )

    r2 = r2_score(
        y_test[:, i],
        test_pred[:, i]
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        "MAE:",
        round(mae, 4),
        "m"
    )

    print(
        "RMSE:",
        round(rmse, 4),
        "m"
    )

    print(
        "R2:",
        round(r2, 4)
    )


# =========================================================
# 2D DISPLACEMENT ERROR
# =========================================================

error_2d = np.sqrt(

    (
        test_pred[:, 0]
        -
        y_test[:, 0]
    ) ** 2

    +

    (
        test_pred[:, 1]
        -
        y_test[:, 1]
    ) ** 2
)


print(
    "\n===== 2D DISPLACEMENT ERROR ====="
)


print(
    "Mean:",
    round(np.mean(error_2d), 4),
    "m"
)


print(
    "Median:",
    round(np.median(error_2d), 4),
    "m"
)


print(
    "P90:",
    round(np.percentile(error_2d, 90), 4),
    "m"
)


print(
    "Maximum:",
    round(np.max(error_2d), 4),
    "m"
)


# =========================================================
# SAVE
# =========================================================

model.save(
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\n===== SAVED =====")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Scaler:",
    SCALER_PATH
)


print(
    "\n" + "=" * 70
)

print(
    "STEP 27 COMPLETE"
)

print(
    "=" * 70
)