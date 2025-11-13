# ==============================
# app.py — Streamlit EMI Prediction & Eligibility App
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Load pre-trained models
# ------------------------------
reg_model = joblib.load("models/xgb_reg_model.pkl")
clf_model = joblib.load("models/xgb_classifier_model.pkl")

st.set_page_config(page_title="EMI Eligibility Predictor", page_icon="💰", layout="wide")

st.title("💰 EMI Prediction & Eligibility Classification App")
st.markdown("Predict your **Maximum EMI** and check your **Eligibility Status** using AI models trained on financial data.")
st.divider()

# ------------------------------
# Feature Engineering Function
# ------------------------------
def apply_feature_engineering(df):
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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


# ------------------------------
# Input Form
# ------------------------------
with st.form("emi_form"):
    st.header("📋 Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional", "Nan"])
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-Employed"])
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, step=0.5)
        company_type = st.selectbox("Company Type", ["Startup", "Small", "Mid-Size", "Large Indian", "Mnc"])
        house_type = st.selectbox("House Type", ["Rented", "Family", "Own"])
        monthly_rent = st.number_input("Monthly Rent", min_value=0.0, step=500.0)
        family_size = st.number_input("Family Size", min_value=1, step=1)

    with col2:
        dependents = st.number_input("Dependents", min_value=0, step=1)
        school_fees = st.number_input("School Fees", min_value=0.0, step=100.0)
        college_fees = st.number_input("College Fees", min_value=0.0, step=100.0)
        travel_expenses = st.number_input("Travel Expenses", min_value=0.0, step=100.0)
        groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0, step=100.0)
        other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0.0, step=100.0)
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
        current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, step=500.0)
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, step=10.0)
        bank_balance = st.number_input("Bank Balance", min_value=0.0, step=500.0)

    with col3:
        emergency_fund = st.number_input("Emergency Fund", min_value=0.0, step=500.0)
        emi_scenario = st.selectbox("EMI Scenario", [
            "Personal Loan EMI",
            "E-commerce Shopping EMI",
            "Education EMI",
            "Vehicle EMI",
            "Home Appliances EMI"
        ])
        requested_amount = st.number_input("Requested Loan Amount", min_value=1000.0, step=1000.0)
        requested_tenure = st.number_input("Requested Tenure (Months)", min_value=6.0, max_value=240.0, step=6.0)
        monthly_salary = st.number_input("Monthly Salary", min_value=0.0, step=500.0)
        max_monthly_emi = st.number_input("Max Monthly EMI", min_value=0.0, step=500.0)

    submitted = st.form_submit_button("🔍 Predict EMI & Eligibility")

# ------------------------------
# On Submit
# ------------------------------
if submitted:
    # Create DataFrame for single record
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "max_monthly_emi": max_monthly_emi
    }])

    # Apply feature engineering (from preprocessing script)
    input_data = apply_feature_engineering(input_data)

    # ------------------------------
    # Make Predictions
    # ------------------------------
    with st.spinner("Predicting... ⏳"):
        predicted_emi = reg_model.predict(input_data)[0]
        eligibility_pred = clf_model.predict(input_data)[0]

    st.success("✅ Prediction Complete!")

    # ------------------------------
    # Display Results
    # ------------------------------
    colA, colB = st.columns(2)
    with colA:
        st.metric("Predicted Maximum EMI", f"₹ {predicted_emi:,.2f}")

    with colB:
        label_mapping = {0: "Not_Eligible", 1: "High_Risk", 2: "Eligible"}
        result = label_mapping.get(eligibility_pred, "Unknown")

        if result == "Eligible":
            st.metric("EMI Eligibility", "🟢 Eligible")
        elif result == "High_Risk":
            st.metric("EMI Eligibility", "🟠 High Risk")
        else:
            st.metric("EMI Eligibility", "🔴 Not Eligible")

    st.divider()
    st.info("💡 Tip: Improve your credit score and reduce expenses to increase eligibility chances.")
