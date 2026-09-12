import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 65)
print("PROJECT PRIME — ML MOTION MODEL V2")
print("=" * 65)

INPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\prime_ml_dataset_v3.csv"

df = pd.read_csv(INPUT_PATH)

print("\n===== DATA =====")
print("Samples:", len(df))
print("Columns:", len(df.columns))

# ---------------------------------------------------------
# TARGETS
# ---------------------------------------------------------

TARGETS = [
    "target_north",
    "target_east"
]

# ---------------------------------------------------------
# REMOVE NON-FEATURE COLUMNS
# ---------------------------------------------------------

DROP_COLUMNS = [
    "target_north",
    "target_east",
    "target_distance",
    "fix_interval_seconds",
    "gps_fix_index"
]

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = df[feature_columns]
y = df[TARGETS]

print("\n===== FEATURES =====")
print("Feature count:", len(feature_columns))

# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ---------------------------------------------------------

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("\n===== CHRONOLOGICAL SPLIT =====")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# ---------------------------------------------------------
# IMPUTATION
# ---------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n===== TRAINING =====")
model.fit(X_train, y_train)

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

val_pred = model.predict(X_val)

print("\n===== VALIDATION RESULTS =====")

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

    print(f"\n{target}")
    print("MAE :", mae, "m")
    print("RMSE:", rmse, "m")
    print("R²  :", r2)

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

test_pred = model.predict(X_test)

print("\n===== FINAL TEST RESULTS =====")

north_error = np.sqrt(
    (test_pred[:, 0] - y_test["target_north"].values) ** 2
    +
    (test_pred[:, 1] - y_test["target_east"].values) ** 2
)

print("\n2D DISPLACEMENT ERROR")
print("Mean   :", np.mean(north_error), "m")
print("Median :", np.median(north_error), "m")
print("P90    :", np.percentile(north_error, 90), "m")
print("Maximum:", np.max(north_error), "m")

# ---------------------------------------------------------
# TARGET DISTANCE COMPARISON
# ---------------------------------------------------------

true_distance = np.sqrt(
    y_test["target_north"].values ** 2 +
    y_test["target_east"].values ** 2
)

pred_distance = np.sqrt(
    test_pred[:, 0] ** 2 +
    test_pred[:, 1] ** 2
)

distance_error = np.abs(
    pred_distance - true_distance
)

print("\n===== DISTANCE ERROR =====")
print("MAE   :", np.mean(distance_error), "m")
print("Median:", np.median(distance_error), "m")
print("P90   :", np.percentile(distance_error, 90), "m")
print("Max   :", np.max(distance_error), "m")

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

MODEL_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_random_forest_v2.pkl"
IMPUTER_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\models\prime_imputer_v2.pkl"

joblib.dump(model, MODEL_PATH)
joblib.dump(imputer, IMPUTER_PATH)

print("\n===== SAVED =====")
print("Model:", MODEL_PATH)
print("Imputer:", IMPUTER_PATH)

print("\n" + "=" * 65)
print("STEP 17 COMPLETE")
print("=" * 65)