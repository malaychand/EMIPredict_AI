# ============================================
# app.py — EMI Prediction & Eligibility App with Model Selection
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="💰 EMI Predictor & Eligibility", layout="wide")

# ===============================
# Model Configuration
# ===============================
MODEL_CONFIG = {
    "XGBoost": {
        "classifier": "models/xgb_classifier_model.pkl",
        "regressor": "models/xgb_reg_model.pkl"
    },
    "Random Forest": {
        "classifier": "models/rf_classifier.pkl",  
        "regressor": "models/rf_regressor.pkl"
    },
    "Logistic/Linear Regression": {
        "classifier": "models/models/logistic_classifier_smote.pkl",
        "regressor": "models/best_linear_regressor.pkl"
    },
}

# ===============================
# Helper Functions
# ===============================
@st.cache_resource
def load_model(path):
    """Load a model from the given path."""
    if not Path(path).exists():
        st.warning(f"⚠️ Model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Failed to load model '{path}': {e}")
        return None


def apply_feature_engineering(df):
    """Apply the same feature engineering logic used in preprocessing."""
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


def safe_predict_regressor(model, X):
    """Safely predict using a regression model."""
    try:
        pred = model.predict(X)
        return float(pred[0])
    except Exception as e:
        st.error(f"⚠️ Regression prediction failed: {e}")
        return None


def safe_predict_classifier(model, X):
    """Safely predict using a classification model."""
    try:
        pred = model.predict(X)
        decoded = None

        # Handle pipelines
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            clf = model.named_steps["classifier"]
        else:
            clf = model

        if hasattr(clf, "classes_"):
            classes = list(clf.classes_)
            if np.issubdtype(np.array(pred).dtype, np.integer):
                decoded = classes[int(pred[0])]
            else:
                decoded = pred[0]
        else:
            decoded = pred[0]

        # Probability (if available)
        proba = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            proba = float(np.max(probs[0]))
        return decoded, proba
    except Exception as e:
        st.error(f"⚠️ Classification prediction failed: {e}")
        return None, None


# ===============================
# Main App
# ===============================
st.title("💰 EMI Prediction & Eligibility Checker")
st.markdown(
    "Select your preferred ML model and enter applicant details to predict "
    "**Maximum EMI** (Regression) and **EMI Eligibility** (Classification)."
)
st.divider()

# ===============================
# Model Selection
# ===============================
st.sidebar.header("🎯 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose ML Algorithm:",
    options=list(MODEL_CONFIG.keys()),
    help="Select the machine learning model for predictions"
)

st.sidebar.info(f"**Current Model:** {selected_model}")
st.sidebar.markdown("---")

# Display model info
with st.sidebar.expander("ℹ️ Model Information"):
    st.markdown(f"""
    **{selected_model}**
    - Classifier: `{MODEL_CONFIG[selected_model]['classifier']}`
    - Regressor: `{MODEL_CONFIG[selected_model]['regressor']}`
    """)

# Load selected models
with st.spinner(f"Loading {selected_model} models..."):
    clf_model = load_model(MODEL_CONFIG[selected_model]["classifier"])
    reg_model = load_model(MODEL_CONFIG[selected_model]["regressor"])

# Check if models loaded successfully
models_loaded = clf_model is not None and reg_model is not None
if not models_loaded:
    st.error(f"⚠️ Could not load {selected_model} models. Please check the file paths.")

# ===============================
# Input Form
# ===============================
with st.form("applicant_form", clear_on_submit=False):
    st.subheader("📝 Applicant Information")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Details**")
        age = st.number_input("Age", 18, 80, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        education = st.selectbox("Education", ["High School", "Graduate", "Postgraduate", "Other"])
        monthly_salary = st.number_input("Monthly Salary (₹)", 0.0, 1000000.0, 30000.0, step=1000.0)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Contract", "Other"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 50.0, 3.0, step=0.5)

    with col2:
        st.markdown("**Housing & Family**")
        company_type = st.selectbox("Company Type", ["Private", "Public", "Government", "Start-up", "Other"])
        house_type = st.selectbox("House Type", ["Owned", "Rented", "Company Provided", "Other"])
        monthly_rent = st.number_input("Monthly Rent (₹)", 0.0, 100000.0, 0.0, step=500.0)
        family_size = st.number_input("Family Size", 1, 15, 3)
        dependents = st.number_input("Dependents", 0, 10, 1)
        school_fees = st.number_input("School Fees (₹)", 0.0, 50000.0, 0.0, step=100.0)
        college_fees = st.number_input("College Fees (₹)", 0.0, 50000.0, 0.0, step=100.0)

    with col3:
        st.markdown("**Financial Details**")
        travel_expenses = st.number_input("Travel Expenses (₹)", 0.0, 50000.0, 1000.0, step=100.0)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0.0, 100000.0, 5000.0, step=100.0)
        other_monthly_expenses = st.number_input("Other Expenses (₹)", 0.0, 50000.0, 1000.0, step=100.0)
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI (₹)", 0.0, 100000.0, 0.0, step=500.0)
        credit_score = st.number_input("Credit Score", 300.0, 900.0, 700.0, step=1.0)
        bank_balance = st.number_input("Bank Balance (₹)", 0.0, 10000000.0, 10000.0, step=500.0)

    st.divider()

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**Loan Request**")
        emergency_fund = st.number_input("Emergency Fund (₹)", 0.0, 1000000.0, 20000.0, step=500.0)
        emi_scenario = st.selectbox("EMI Scenario", ["Home Loan", "Personal Loan", "Car Loan", "Education Loan", "Other"])
        requested_amount = st.number_input("Requested Loan Amount (₹)", 1000.0, 10000000.0, 500000.0, step=1000.0)
    with col5:
        st.markdown("**Tenure**")
        requested_tenure = st.number_input("Requested Tenure (months)", 1, 360, 60)

    submitted = st.form_submit_button("🔍 Predict EMI & Eligibility", use_container_width=True)

# ===============================
# Prediction Section
# ===============================
if submitted:
    if not models_loaded:
        st.error("❌ Cannot make predictions. Models not loaded properly.")
    else:
        # Prepare input data
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
            "requested_tenure": requested_tenure
        }])

        # Apply feature engineering
        input_data = apply_feature_engineering(input_data)

        # Show input summary in expander
        with st.expander("📊 View Input Summary (with Engineered Features)", expanded=False):
            st.dataframe(input_data.T.rename(columns={0: "Value"}), height=420)

        st.divider()
        
        # Display predictions
        st.subheader(f"🎯 Predictions using {selected_model}")
        
        colA, colB = st.columns(2)

        # --- Regression Prediction ---
        with colA:
            st.markdown("### 📈 Maximum EMI Prediction")
            if reg_model:
                pred_reg = safe_predict_regressor(reg_model, input_data)
                if pred_reg is not None:
                    st.metric(
                        label="Predicted Maximum EMI",
                        value=f"₹ {pred_reg:,.2f}"
                    )
                    st.success("✅ Prediction successful")
            else:
                st.error("❌ Regression model not loaded.")

        # --- Classification Prediction ---
        with colB:
            st.markdown("### 🎯 EMI Eligibility Classification")
            if clf_model:
                pred_cls, cls_proba = safe_predict_classifier(clf_model, input_data)
                label_map = {0: "Not_Eligible", 1: "High_Risk", 2: "Eligible"}
                label = label_map.get(pred_cls, str(pred_cls))
                
                if pred_cls is not None:
                    if label == "Eligible":
                        st.success(f"✅ **Status:** {label}")
                    elif label == "High_Risk":
                        st.warning(f"⚠️ **Status:** {label}")
                    else:
                        st.error(f"🚫 **Status:** {label}")
                    
                    if cls_proba:
                        st.metric(
                            label="Confidence Level",
                            value=f"{cls_proba:.2%}"
                        )
            else:
                st.error("❌ Classifier model not loaded.")

        st.divider()
        st.caption(f"💡 **Note:** Predictions made using **{selected_model}** algorithm. "
                   "Models include preprocessing (scaling, encoding). Feature engineering is applied automatically.")