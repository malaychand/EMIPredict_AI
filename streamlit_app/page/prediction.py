# streamlit_app/page/prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
def show_prediction(MODEL_DIR="models"):
    # -----------------------------
    # Project root (EMI_Predict_AI)
    # -----------------------------
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(ROOT / "scripts"))
    MODEL_DIR = ROOT / "models"
    PREPROC_DIR = MODEL_DIR / "preprocessors"

    # -----------------------------
    # Import prediction function
    # -----------------------------
    from predict_emi import predict_emi

    # -----------------------------
    # Cache models, encoders, features
    # -----------------------------
    @st.cache_resource
    def load_label_encoder():
        return joblib.load(PREPROC_DIR / "eligibility_label_encoder.joblib")

    @st.cache_resource
    def load_clf_features():
        return joblib.load(MODEL_DIR / "clf_features.joblib")

    @st.cache_resource
    def load_reg_features():
        return joblib.load(MODEL_DIR / "reg_features.joblib")

    label_encoder = load_label_encoder()
    clf_features = load_clf_features()
    reg_features = load_reg_features()

    # -----------------------------
    # Streamlit UI
    # -----------------------------
    st.title("💰 EMI Eligibility & Prediction")
    st.markdown("### Enter Applicant Details")

    # -----------------------------
    # Applicant details
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 18, 75, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        employment_type = st.selectbox("Employment Type", ["Government", "Private", "Self-employed"])
        company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Startup", "Large Indian", "Small"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 50.0, 3.0, 0.1)
        monthly_salary = st.number_input("Monthly Salary (₹)", 1000.0, 1_000_000.0, 40000.0, 1000.0)

    with col2:
        family_size = st.number_input("Family Size", 1, 20, 4)
        dependents = st.number_input("Dependents", 0, 10, 2)
        house_type = st.selectbox("House Type", ["Own", "Family", "Rented"])
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
        current_emi_amount = st.number_input("Current EMI Amount (₹)", 0.0, 500_000.0, 5000.0, 500.0)
        requested_amount = st.number_input("Requested Loan Amount (₹)", 10000.0, 5_000_000.0, 100000.0, 5000.0)
        requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 24)
        emi_scenario = st.selectbox(
            "EMI Scenario",
            ["Personal Loan EMI", "E-commerce Shopping EMI", "Education EMI", "Vehicle EMI", "Home Appliances EMI"]
        )

    # -----------------------------
    # Financial stability
    # -----------------------------
    st.markdown("### 💳 Financial Stability Details")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
    with colf2:
        bank_balance = st.number_input("Bank Balance (₹)", 0.0, 10_000_000.0, 100000.0, 1000.0)
    with colf3:
        emergency_fund = st.number_input("Emergency Fund (₹)", 0.0, 10_000_000.0, 50000.0, 1000.0)

    # -----------------------------
    # Monthly expenses
    # -----------------------------
    st.markdown("### Enter Monthly Expense Details")
    colA, colB, colC = st.columns(3)
    with colA:
        school_fees = st.number_input("School Fees (₹)", 0.0, 100_000.0, 2000.0, 500.0)
        college_fees = st.number_input("College Fees (₹)", 0.0, 500_000.0, 3000.0, 500.0)
    with colB:
        travel_expenses = st.number_input("Travel Expenses (₹)", 0.0, 50_000.0, 1500.0, 500.0)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0.0, 50_000.0, 5000.0, 500.0)
    with colC:
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0.0, 50_000.0, 1000.0, 500.0)
        monthly_rent = st.number_input("Monthly Rent (₹)", 0.0, 200_000.0, 8000.0, 500.0)

    # -----------------------------
    # Combine inputs
    # -----------------------------
    input_dict = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "company_type": company_type,
        "years_of_employment": years_of_employment,
        "monthly_salary": monthly_salary,
        "family_size": family_size,
        "dependents": dependents,
        "house_type": house_type,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "emi_scenario": emi_scenario,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "monthly_rent": monthly_rent
    }


    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔍 Predict EMI"):
        with st.spinner("Running prediction..."):
            try:
                clf_pred, reg_pred = predict_emi(input_dict, MODEL_DIR)

                st.success("✅ Prediction Complete")

                eligibility_badge = {
                    "Eligible": "🟢 Eligible",
                    "Not_Eligible": "🔴 Not Eligible",
                    "High_Risk": "🟠 High Risk"
                }
                st.markdown(f"### **EMI Eligibility:** {eligibility_badge.get(clf_pred, clf_pred)}")
                st.markdown(f"### **Predicted Max Monthly EMI:** ₹ {reg_pred:,.2f}")

            except FileNotFoundError as e:
                st.error(f"⚠️ Model or preprocessor file not found: {e}")
            except KeyError as e:
                st.error(f"⚠️ Missing feature for prediction: {e}")
            except ValueError as e:
                st.error(f"⚠️ Value error: {e}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")
