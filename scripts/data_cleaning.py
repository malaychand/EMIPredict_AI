"""
DATA CLEANING SCRIPT (FINAL)
----------------------------
Cleans and prepares raw EMI dataset for model training.

Order of Operations:
1. Load dataset
2. Remove duplicates
3. Clean key categorical/numeric columns
4. Clean numeric-like strings (e.g. "303200.0.0")
5. Handle missing values (features only)
6. Handle numeric outliers (IQR capping)
7. Optional log-transform for skewed numeric features
8. Save cleaned dataset to artifacts/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "emi_prediction_dataset.csv"
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print(f"📥 Loading dataset: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)
print(f"Initial shape: {df.shape}")

# -----------------------------
# 2. REMOVE DUPLICATES
# -----------------------------
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"✅ Removed {before - after} duplicate rows. New shape: {df.shape}")

# -----------------------------
# 3. CLEAN KEY COLUMNS
# -----------------------------
# --- AGE ---
df["age"] = df["age"].astype(str).str.extract(r"(\d+)")
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# --- GENDER ---
df["gender"] = df["gender"].astype(str).str.lower().str.strip()
gender_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
df["gender"] = df["gender"].map(gender_map)
print("Unique genders after cleaning:", df["gender"].unique())

# --- MARITAL STATUS ---
df["marital_status"] = df["marital_status"].astype(str).str.title().str.strip()

# --- EDUCATION ---
df["education"] = df["education"].astype(str).str.title().str.strip()

# --- COMPANY TYPE, HOUSE TYPE, EMPLOYMENT TYPE ---
for col in ["company_type", "house_type", "employment_type"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.title().str.strip()

# --- EXISTING LOANS ---
df["existing_loans"] = df["existing_loans"].astype(str).str.title().str.strip()

# -----------------------------
# 4. CLEAN NUMERIC-LIKE STRINGS
# -----------------------------
def clean_numeric(val):
    try:
        val = str(val).replace(".0.0", "").replace(".0", "")
        return float(val)
    except:
        return np.nan

numeric_cols = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# -----------------------------
# 4b. SAFE INTEGER CONVERSION (ROUNDING FLOATS)
# -----------------------------
int_cols = ["age", "family_size", "dependents", "requested_tenure"]
for col in int_cols:
    if col in df.columns:
        # Round floats to nearest integer and convert to nullable Int64
        df[col] = df[col].round(0).astype("Int64")

# -----------------------------
# 5. HANDLE MISSING VALUES
# -----------------------------
target_cols = ["emi_eligibility", "max_monthly_emi"]
feature_cols = [c for c in df.columns if c not in target_cols]

num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns

# Fill numeric columns with median
for col in num_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in cat_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing values handled.")

# -----------------------------
# 6. HANDLE OUTLIERS (IQR)
# -----------------------------
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

print("✅ IQR-based outlier capping applied to numeric features.")

# -----------------------------
# 7. OPTIONAL: LOG TRANSFORM
# -----------------------------
skewed_cols = [
    "monthly_salary", "monthly_rent", "college_fees",
    "emergency_fund", "requested_amount", "current_emi_amount"
]

for col in skewed_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])
print("✅ Log1p transform applied to skewed features.")

# -----------------------------
# 8. SAVE CLEANED DATA
# -----------------------------

clean_path = OUTPUT_DIR / "cleaned_EMI_dataset.csv"
df.to_csv(clean_path, index=False)

print("\n🎯 POST-CLEANING CHECK")
print(df.info())
print(df.isnull().sum().sum(), "total missing values remain.")
print(f"✅ Cleaned dataset saved to: {clean_path}")

# -----------------------------
# 9. PRINT UNIQUE CATEGORICAL VALUES
# -----------------------------
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
unique_keys_dict = {col: df[col].unique().tolist() for col in categorical_cols}

print("\n--- UNIQUE KEYS (CATEGORICAL COLUMNS) ---")
for col, keys in unique_keys_dict.items():
    print(f"{col}: {keys}")
