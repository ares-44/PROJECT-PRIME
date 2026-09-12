import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("PROJECT PRIME — VELOCITY ML MODEL V1")
print("=" * 70)


# =========================================================
# PATHS
# =========================================================

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_velocity_dataset_v1.csv"

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_velocity_rf_v1.pkl"

IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_velocity_imputer_v1.pkl"


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# TARGETS
# =========================================================

TARGETS = [
    "target_velocity_north",
    "target_velocity_east"
]


# =========================================================
# REMOVE TARGET / METADATA
# =========================================================

DROP_COLUMNS = [
    "target_velocity_north",
    "target_velocity_east",
    "target_velocity",
    "interval_seconds",
    "gps_fix_index"
]

FEATURE_COLUMNS = [
    col
    for col in df.columns
    if col not in DROP_COLUMNS
]

X = df[FEATURE_COLUMNS]
y = df[TARGETS]


print("\n===== FEATURES =====")
print("Feature count:", len(FEATURE_COLUMNS))


# =========================================================
# CHRONOLOGICAL SPLIT
# =========================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]


print("\n===== SPLIT =====")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# =========================================================
# IMPUTATION
# =========================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_val = imputer.transform(
    X_val
)

X_test = imputer.transform(
    X_test
)


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


print("\n===== TRAINING =====")

model.fit(
    X_train,
    y_train
)


# =========================================================
# VALIDATION
# =========================================================

val_pred = model.predict(
    X_val
)


print("\n===== VALIDATION =====")


for i, target in enumerate(TARGETS):

    mae = mean_absolute_error(
        y_val.iloc[:, i],
        val_pred[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val.iloc[:, i],
            val_pred[:, i]
        )
    )

    r2 = r2_score(
        y_val.iloc[:, i],
        val_pred[:, i]
    )

    print("\n", target)

    print(
        "MAE:",
        round(mae, 4),
        "m/s"
    )

    print(
        "RMSE:",
        round(rmse, 4),
        "m/s"
    )

    print(
        "R2:",
        round(r2, 4)
    )


# =========================================================
# FINAL TEST
# =========================================================

test_pred = model.predict(
    X_test
)


print("\n===== FINAL TEST =====")


for i, target in enumerate(TARGETS):

    mae = mean_absolute_error(
        y_test.iloc[:, i],
        test_pred[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test.iloc[:, i],
            test_pred[:, i]
        )
    )

    r2 = r2_score(
        y_test.iloc[:, i],
        test_pred[:, i]
    )

    print("\n", target)

    print(
        "MAE:",
        round(mae, 4),
        "m/s"
    )

    print(
        "RMSE:",
        round(rmse, 4),
        "m/s"
    )

    print(
        "R2:",
        round(r2, 4)
    )


# =========================================================
# SPEED ERROR
# =========================================================

true_speed = np.sqrt(
    y_test["target_velocity_north"].values ** 2
    +
    y_test["target_velocity_east"].values ** 2
)

pred_speed = np.sqrt(
    test_pred[:, 0] ** 2
    +
    test_pred[:, 1] ** 2
)

speed_error = np.abs(
    pred_speed - true_speed
)


print("\n===== SPEED ERROR =====")

print(
    "MAE:",
    round(np.mean(speed_error), 4),
    "m/s"
)

print(
    "Median:",
    round(np.median(speed_error), 4),
    "m/s"
)

print(
    "P90:",
    round(np.percentile(speed_error, 90), 4),
    "m/s"
)

print(
    "Maximum:",
    round(np.max(speed_error), 4),
    "m/s"
)


# =========================================================
# SAVE
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    imputer,
    IMPUTER_PATH
)


print("\n===== SAVED =====")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Imputer:",
    IMPUTER_PATH
)


print(
    "\n" + "=" * 70
)

print(
    "STEP 24 COMPLETE"
)

print(
    "=" * 70
)