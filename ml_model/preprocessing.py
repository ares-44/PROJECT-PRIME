import pandas as pd

DATA_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\IO-VNBD-GIT\Synchronised V abd S datasets\Categorised IOVNB Dataset\S (Driver A)\S1\S-S1.csv"

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Clean column names
# --------------------------------------------------

df.columns = df.columns.str.strip()

# --------------------------------------------------
# 2. Convert timestamp
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["DATE (YYYY-MO-DD HH-MI-SS_SSS)"].str.replace(
        r"(\d{2}:\d{2}:\d{2}):(\d{3})$",
        r"\1.\2",
        regex=True
    ),
    errors="coerce"
)


# --------------------------------------------------
# 3. Convert GPS satellite information
# Example: "27 / 28" -> 27
# --------------------------------------------------

df["GPS SATELLITES"] = (
    df["GPS SATELLITES IN RANGE"]
    .str.extract(r"(\d+)")
    .astype(float)
)

# --------------------------------------------------
# 4. Remove original text columns
# --------------------------------------------------

df = df.drop(columns=[
    "GPS SATELLITES IN RANGE",
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
])

# --------------------------------------------------
# 5. Check result
# --------------------------------------------------

print("\n===== CLEAN DATASET =====")
print("Shape:", df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== FIRST 5 ROWS =====")
print(df.head())
OUTPUT_PATH = r"C:\Users\Adarsh Kumar Pal\OneDrive\Desktop\PROJECT_PRIME\data\cleaned_S-S1.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\n===== SAVED =====")
print("Cleaned dataset saved to:")
print(OUTPUT_PATH)