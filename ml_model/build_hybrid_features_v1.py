import os
import numpy as np
import pandas as pd

print("=" * 75)
print("PROJECT PRIME — HYBRID PHYSICS FEATURES V1")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SEQUENCE_DATA = os.path.join(
    BASE,
    "data",
    "prime_sequence_dataset_v2.npz"
)

OUT = os.path.join(
    BASE,
    "data",
    "prime_hybrid_dataset_v1.npz"
)


# ============================================================
# LOAD SEQUENCE DATA
# ============================================================

data = np.load(
    SEQUENCE_DATA,
    allow_pickle=True
)

X = data["X"]
y = data["y"]
intervals = data["intervals"]
fix_ids = data["fix_ids"]
sensor_columns = data["sensor_columns"]


print("\n===== DATA =====")

print(
    "Sequence shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)

print(
    "Samples:",
    len(X)
)


# ============================================================
# SENSOR COLUMN INDICES
# ============================================================

sensor_columns = list(
    sensor_columns
)

print("\n===== SENSOR CHANNELS =====")

for i, column in enumerate(
    sensor_columns
):

    print(
        f"{i + 1:02d}. {column}"
    )


def find_column(
    keyword
):

    for i, column in enumerate(
        sensor_columns
    ):

        if keyword.lower() in column.lower():

            return i

    raise ValueError(
        f"Could not find sensor column: {keyword}"
    )


acc_x = find_column(
    "ACCELEROMETER X"
)

acc_y = find_column(
    "ACCELEROMETER Y"
)

acc_z = find_column(
    "ACCELEROMETER Z"
)

grav_x = find_column(
    "GRAVITY X"
)

grav_y = find_column(
    "GRAVITY Y"
)

grav_z = find_column(
    "GRAVITY Z"
)

gyro_yaw = find_column(
    "GYROSCOPE Yaw"
)

gyro_pitch = find_column(
    "GYROSCOPE Pitch"
)

gyro_roll = find_column(
    "GYROSCOPE Roll"
)


# ============================================================
# SAMPLING INTERVAL
# ============================================================

# Dataset is approximately 10 Hz.

DT = 0.1

print(
    "\nSampling interval:",
    DT,
    "seconds"
)


# ============================================================
# BUILD PHYSICS FEATURES
# ============================================================

physics_features = []


for sequence in X:

    # --------------------------------------------------------
    # Sensor-frame acceleration
    # --------------------------------------------------------

    ax = sequence[:, acc_x]

    ay = sequence[:, acc_y]

    az = sequence[:, acc_z]


    gx = sequence[:, grav_x]

    gy = sequence[:, grav_y]

    gz = sequence[:, grav_z]


    # Gravity removed acceleration

    lax = ax - gx

    lay = ay - gy

    laz = az - gz


    # --------------------------------------------------------
    # Acceleration magnitude
    # --------------------------------------------------------

    linear_acc_mag = np.sqrt(
        lax ** 2
        +
        lay ** 2
        +
        laz ** 2
    )


    # --------------------------------------------------------
    # Velocity integration
    # --------------------------------------------------------

    vx = np.cumsum(
        lax
    ) * DT

    vy = np.cumsum(
        lay
    ) * DT

    vz = np.cumsum(
        laz
    ) * DT


    # --------------------------------------------------------
    # Displacement integration
    #
    # Simple sensor-frame integration.
    # No orientation correction is assumed.
    # --------------------------------------------------------

    dx = np.cumsum(
        vx
    ) * DT

    dy = np.cumsum(
        vy
    ) * DT

    dz = np.cumsum(
        vz
    ) * DT


    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    yaw_rate = sequence[
        :,
        gyro_yaw
    ]

    pitch_rate = sequence[
        :,
        gyro_pitch
    ]

    roll_rate = sequence[
        :,
        gyro_roll
    ]


    gyro_mag = np.sqrt(
        yaw_rate ** 2
        +
        pitch_rate ** 2
        +
        roll_rate ** 2
    )


    # --------------------------------------------------------
    # Angular change over window
    # --------------------------------------------------------

    yaw_change = (
        np.sum(yaw_rate)
        * DT
    )

    pitch_change = (
        np.sum(pitch_rate)
        * DT
    )

    roll_change = (
        np.sum(roll_rate)
        * DT
    )


    # --------------------------------------------------------
    # Final physics state
    # --------------------------------------------------------

    features = [

        # Final velocity
        vx[-1],
        vy[-1],
        vz[-1],

        # Final displacement
        dx[-1],
        dy[-1],
        dz[-1],

        # Magnitude of integrated sensor-frame displacement
        np.sqrt(
            dx[-1] ** 2
            +
            dy[-1] ** 2
            +
            dz[-1] ** 2
        ),

        # Mean linear acceleration magnitude
        np.mean(
            linear_acc_mag
        ),

        # Maximum linear acceleration magnitude
        np.max(
            linear_acc_mag
        ),

        # Final velocity magnitude
        np.sqrt(
            vx[-1] ** 2
            +
            vy[-1] ** 2
            +
            vz[-1] ** 2
        ),

        # Gyro angular changes
        yaw_change,
        pitch_change,
        roll_change,

        # Mean gyro magnitude
        np.mean(
            gyro_mag
        ),

        # Maximum gyro magnitude
        np.max(
            gyro_mag
        )
    ]


    physics_features.append(
        features
    )


physics_features = np.asarray(
    physics_features,
    dtype=np.float32
)


# ============================================================
# CHECK
# ============================================================

print(
    "\n===== PHYSICS FEATURES ====="
)

print(
    "Shape:",
    physics_features.shape
)

print(
    "Physics features:",
    physics_features.shape[1]
)


feature_names = np.array(
    [
        "final_velocity_x",
        "final_velocity_y",
        "final_velocity_z",

        "integrated_displacement_x",
        "integrated_displacement_y",
        "integrated_displacement_z",

        "integrated_displacement_magnitude",

        "mean_linear_acceleration",
        "max_linear_acceleration",

        "final_velocity_magnitude",

        "yaw_change",
        "pitch_change",
        "roll_change",

        "mean_gyro_magnitude",
        "max_gyro_magnitude"
    ]
)


# ============================================================
# STATISTICS
# ============================================================

print(
    "\n===== PHYSICS FEATURE STATISTICS ====="
)

for i, name in enumerate(
    feature_names
):

    values = physics_features[
        :,
        i
    ]

    print(
        f"{name:35s}"
        f" mean={np.mean(values):10.4f}"
        f" std={np.std(values):10.4f}"
    )


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if np.isnan(
    physics_features
).any():

    raise ValueError(
        "NaN detected in physics features."
    )


if np.isinf(
    physics_features
).any():

    raise ValueError(
        "Infinite value detected in physics features."
    )


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUT,

    X=X,

    physics_features=
        physics_features,

    y=y,

    intervals=
        intervals,

    fix_ids=
        fix_ids,

    sensor_columns=
        np.array(
            sensor_columns
        ),

    physics_feature_names=
        feature_names
)


print(
    "\n===== SAVED ====="
)

print(
    OUT
)


print(
    "\n" + "=" * 75
)

print(
    "STEP 34A COMPLETE"
)

print(
    "=" * 75
)