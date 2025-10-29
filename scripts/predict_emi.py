# scripts/predict_emi.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def predict_emi(input_dict, model_dir):
    """
    Predict EMI eligibility (classification) and max monthly EMI (regression)
    using trained models and feature-engineered inputs.
    """

    df_input = pd.DataFrame([input_dict])
    model_dir = Path(model_dir)

    # -----------------------------
    # Load models and preprocessors
    # -----------------------------
    clf_model = joblib.load(model_dir / "best_classifier.joblib")
    reg_model = joblib.load(model_dir / "best_regressor.joblib")
    label_encoder = joblib.load(model_dir / "preprocessors" / "eligibility_label_encoder.joblib")
    target_scaler = joblib.load(model_dir / "preprocessors" / "target_scaler.joblib")
    clf_features = joblib.load(model_dir / "clf_features.joblib")
    reg_features = joblib.load(model_dir / "reg_features.joblib")
    scaler = joblib.load(model_dir / "preprocessors" / "scaler.joblib")

    # Load label encoders for categorical features
    encoders_path = model_dir / "preprocessors" / "label_encoders.joblib"
    if encoders_path.exists():
        label_encoders = joblib.load(encoders_path)
    else:
        label_encoders = {}

    # -----------------------------
    # Encode categorical columns safely
    # -----------------------------
    for col, le in label_encoders.items():
        if col in df_input.columns:
            try:
                df_input[col] = le.transform(df_input[col])
            except ValueError:
                # unseen category → assign most frequent class (first class)
                df_input[col] = [le.transform([le.classes_[0]])[0]]

    # -----------------------------
    # Log-transform skewed numeric inputs
    # -----------------------------
    skewed_cols = [
        "monthly_salary", "monthly_rent", "college_fees",
        "emergency_fund", "requested_amount", "current_emi_amount"
    ]
    for col in skewed_cols:
        if col in df_input.columns:
            df_input[col] = np.log1p(df_input[col])

    # -----------------------------
    # Derived feature engineering
    # -----------------------------
    expense_cols = [
        "school_fees", "college_fees", "travel_expenses",
        "groceries_utilities", "other_monthly_expenses", "monthly_rent"
    ]

    df_input["total_monthly_expenses"] = df_input[expense_cols].sum(axis=1)
    df_input["debt_to_income"] = df_input["current_emi_amount"] / np.maximum(df_input["monthly_salary"], 1)
    df_input["expense_to_income"] = df_input["total_monthly_expenses"] / np.maximum(df_input["monthly_salary"], 1)
    df_input["monthly_disposable"] = df_input["monthly_salary"] - df_input["total_monthly_expenses"] - df_input["current_emi_amount"]
    df_input["instalment_if_approved"] = df_input["requested_amount"] / np.maximum(df_input["requested_tenure"], 1)
    df_input["affordability_ratio"] = df_input["monthly_disposable"] / np.maximum(df_input["instalment_if_approved"], 1)
    df_input["employment_stability"] = df_input["years_of_employment"] / np.maximum(df_input["age"], 1)
    df_input["loan_to_income_ratio"] = df_input["requested_amount"] / np.maximum(df_input["monthly_salary"], 1)
    df_input["dependents_ratio"] = df_input["dependents"] / np.maximum(df_input["family_size"], 1)

    # -----------------------------
    # Scale numeric + derived features
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

    numeric_cols = [c for c in numeric_cols if c in df_input.columns]
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    # -----------------------------
    # Prepare features for classification
    # -----------------------------
    for col in clf_features:
        if col not in df_input.columns:
            df_input[col] = 0

    X_clf = df_input[clf_features].values

    # -----------------------------
    # Classification prediction
    # -----------------------------
    clf_pred_num = clf_model.predict(X_clf)[0]
    try:
        clf_pred_label = label_encoder.inverse_transform([int(clf_pred_num)])[0]
    except Exception:
        clf_pred_label = str(clf_pred_num)

    # -----------------------------
    # Prepare regression features
    # -----------------------------
    X_reg = df_input.copy()
    if "emi_eligibility" in reg_features:
        X_reg["emi_eligibility"] = label_encoder.transform([clf_pred_label])[0]

    for col in reg_features:
        if col not in X_reg.columns:
            X_reg[col] = 0

    X_reg = X_reg[reg_features].values

    # -----------------------------
    # Regression prediction
    # -----------------------------
    reg_pred_scaled = reg_model.predict(X_reg)[0]
    reg_pred = target_scaler.inverse_transform([[reg_pred_scaled]])[0][0]

    # Ensure positive, rounded output
    reg_pred = max(0, round(reg_pred, 2))

    return clf_pred_label, reg_pred
