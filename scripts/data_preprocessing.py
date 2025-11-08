# data_preprocessing.py

import os
import pandas as pd
import numpy as np

def load_and_preprocess_data(data_path):
    """
    Load and preprocess EMI dataset for regression.
    Performs feature engineering and data cleaning.
    """
    print(f"📥 Loading dataset from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with shape: {df.shape}")

    # Feature Engineering
    df["debt_to_income"] = df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)
    
    expense_cols = [
        "school_fees", "college_fees", "travel_expenses",
        "groceries_utilities", "other_monthly_expenses", "monthly_rent"
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

    # Data Cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['max_monthly_emi'], inplace=True)
    df.dropna(inplace=True)

    print(f"🧹 Cleaned dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    test_path = "data/cleaned_EMI_dataset.csv"
    df = load_and_preprocess_data(test_path)
    print(df.head())
