# scripts/feature_engineering.py

"""
FEATURE ENGINEERING SCRIPT (UPDATED)
---------------------------------
Tasks:
1. Load cleaned EMI dataset
2. Create derived financial features
3. Encode categorical variables
4. Encode and scale targets (emi_eligibility & max_monthly_emi)
5. Scale numeric features
6. Save feature-engineered dataset and preprocessors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# CONFIGURATION
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "artifacts" / "cleaned_EMI_dataset.csv"
OUTPUT_DIR = ROOT / "artifacts"
PREPROC_DIR = ROOT / "models" / "preprocessors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREPROC_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. LOAD CLEANED DATA
# -----------------------------
print(f"📥 Loading cleaned dataset: {CLEAN_FILE}")
df = pd.read_csv(CLEAN_FILE)
print(f"Initial shape: {df.shape}")

# -----------------------------
# 2. DERIVED FEATURES
# -----------------------------
df["debt_to_income"] = df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)

expense_cols = [
    "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
    "other_monthly_expenses", "monthly_rent"
]
df["total_monthly_expenses"] = df[expense_cols].sum(axis=1)
df["expense_to_income"] = df["total_monthly_expenses"] / df["monthly_salary"].replace(0, np.nan)
df["monthly_disposable"] = (
    df["monthly_salary"] - df["total_monthly_expenses"] - df["current_emi_amount"]
)
df["instalment_if_approved"] = df["requested_amount"] / df["requested_tenure"].replace(0, np.nan)
df["affordability_ratio"] = df["monthly_disposable"] / df["instalment_if_approved"].replace(0, np.nan)
df["employment_stability"] = df["years_of_employment"] / df["age"].replace(0, np.nan)
df["loan_to_income_ratio"] = df["requested_amount"] / df["monthly_salary"].replace(0, np.nan)
df["dependents_ratio"] = df["dependents"] / df["family_size"].replace(0, np.nan)

# Clean infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# -----------------------------
# 3. SEPARATE TARGETS
# -----------------------------
target_class = "emi_eligibility"
target_reg = "max_monthly_emi"

y_class = df[target_class]
y_reg = df[target_reg]
df = df.drop([target_class, target_reg], axis=1)

# -----------------------------
# 4. CATEGORICAL ENCODING
# -----------------------------
categorical_cols = [
    "gender", "marital_status", "education", "employment_type",
    "house_type", "company_type", "emi_scenario", "existing_loans"
]

label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Encode classification target separately
clf_encoder = LabelEncoder()
y_class_encoded = clf_encoder.fit_transform(y_class)
label_encoders[target_class] = clf_encoder

# Save all label encoders
joblib.dump(label_encoders, PREPROC_DIR / "label_encoders.joblib")
joblib.dump(clf_encoder, PREPROC_DIR / "eligibility_label_encoder.joblib")
print("✅ Label encoders saved.")

# -----------------------------
# 5. NUMERIC SCALING (FEATURES)
# -----------------------------
numeric_cols = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
    "debt_to_income", "expense_to_income", "affordability_ratio",
    "monthly_disposable", "instalment_if_approved", "employment_stability",
    "loan_to_income_ratio", "dependents_ratio"
]

numeric_cols = [c for c in numeric_cols if c in df.columns]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save feature scaler
joblib.dump(scaler, PREPROC_DIR / "scaler.joblib")
print("✅ Feature scaler saved.")

# -----------------------------
# 6. SCALE REGRESSION TARGET
# -----------------------------
target_scaler = StandardScaler()
y_reg_scaled = target_scaler.fit_transform(y_reg.values.reshape(-1, 1))
joblib.dump(target_scaler, PREPROC_DIR / "target_scaler.joblib")
print("✅ Target scaler saved for max_monthly_emi")

# -----------------------------
# 7. MERGE BACK TARGETS & SAVE
# -----------------------------
df[target_class] = y_class_encoded
df[target_reg] = y_reg_scaled

FEATURE_FILE = OUTPUT_DIR / "feature_engineered_EMI_dataset.csv"
df.to_csv(FEATURE_FILE, index=False)

print(f"\n🎯 Feature-engineered dataset saved to: {FEATURE_FILE}")
print(f"Final shape: {df.shape}")
