# app.py — EMI Prediction & Eligibility (updated)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

st.set_page_config(page_title="💰 EMI Predictor & Eligibility", layout="wide")

# -----------------------
# MODEL CONFIG (edit paths if needed)
# -----------------------
MODEL_CONFIG = {
    "XGBoost": {
        "classifier": "models/xgb_classifier.pkl",
        "regressor": "models/xgb_regressor.pkl"
    },
    "Random Forest": {
        "classifier": "models/rf_classifier.pkl",
        "regressor": "models/rf_regressor.pkl"
    },
    "Logistic/Linear": {
        # These names match your prior notebook's joblib.dump filenames
        "classifier": "models/logistic_classifier.pkl",
        "regressor": "models/linear_regressor.pkl"
    },
}

# -----------------------
# Utilities
# -----------------------
@st.cache_resource
def load_model(path: str):
    """Load model with caching. Returns None on failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model at `{path}` — {e}")
        return None

def df_from_form(values: Dict[str, Any]) -> pd.DataFrame:
    """Create a single-row DataFrame from form inputs (values is a dict)."""
    return pd.DataFrame([values])

def to_download_button(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """Streamlit download button for a dataframe CSV."""
    csv = df.to_csv(index=False)
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# -----------------------
# Feature engineering (based on your preprocessing)
# (unchanged — using your robust version)
# -----------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    expense_cols = [
        "school_fees", "college_fees", "travel_expenses",
        "groceries_utilities", "other_monthly_expenses", "monthly_rent"
    ]
    for col in expense_cols:
        if col not in df.columns:
            df[col] = 0.0

    numeric_candidates = [
        "monthly_salary", "current_emi_amount", "requested_amount",
        "requested_tenure", "years_of_employment", "age",
        "dependents", "family_size", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "monthly_rent", "credit_score", "bank_balance", "emergency_fund"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df["debt_to_income"] = np.where(
        df.get("monthly_salary", 0) != 0,
        df.get("current_emi_amount", 0) / df.get("monthly_salary", np.nan),
        np.nan
    )

    df["total_monthly_expenses"] = df[expense_cols].sum(axis=1)

    df["expense_to_income"] = np.where(
        df.get("monthly_salary", 0) != 0,
        df["total_monthly_expenses"] / df.get("monthly_salary", np.nan),
        np.nan
    )

    # safe subtraction with fill
    df["monthly_disposable"] = (
        df.get("monthly_salary", pd.Series(0,index=df.index)).fillna(0)
        - df["total_monthly_expenses"].fillna(0)
        - df.get("current_emi_amount", pd.Series(0,index=df.index)).fillna(0)
    )

    df["instalment_if_approved"] = np.where(
        (df.get("requested_tenure", 0) != 0),
        df.get("requested_amount", 0) / df.get("requested_tenure", np.nan),
        np.nan
    )

    df["affordability_ratio"] = np.where(
        df.get("instalment_if_approved", 0) != 0,
        df["monthly_disposable"] / df.get("instalment_if_approved", np.nan),
        np.nan
    )

    df["employment_stability"] = np.where(
        df.get("age", 0) != 0,
        df.get("years_of_employment", 0) / df.get("age", np.nan),
        np.nan
    )

    df["loan_to_income_ratio"] = np.where(
        df.get("monthly_salary", 0) != 0,
        df.get("requested_amount", 0) / df.get("monthly_salary", np.nan),
        np.nan
    )

    df["dependents_ratio"] = np.where(
        df.get("family_size", 0) != 0,
        df.get("dependents", 0) / df.get("family_size", np.nan),
        np.nan
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    engineered_cols_zero_fill = [
        "debt_to_income", "expense_to_income", "instalment_if_approved",
        "affordability_ratio", "employment_stability", "loan_to_income_ratio", "dependents_ratio"
    ]
    for col in engineered_cols_zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median(skipna=True)
            if np.isnan(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    return df

# -----------------------
# Prediction helpers (enhanced)
# -----------------------
def model_predict_regression(model, X: pd.DataFrame):
    """Return scalar prediction for regression models. Works with pipelines and raw models."""
    try:
        pred = model.predict(X)
        return float(np.asarray(pred).ravel()[0]) if pred is not None else None
    except Exception:
        try:
            arr = X.values
            pred = model.predict(arr)
            return float(np.asarray(pred).ravel()[0])
        except Exception as e2:
            st.error(f"Regression prediction failed: {e2}")
            return None

def model_predict_classification(model, X: pd.DataFrame):
    """
    Return (pred_label_string, confidence_float, raw_pred_value).
    Heuristics:
     - If the model/pipeline contains a classifier with classes_ that are strings, map back.
     - If numeric (0/1) with no mapping, map 1->'Eligible', 0->'Not Eligible'.
    """
    try:
        pred = model.predict(X)
        raw = np.asarray(pred).ravel()[0]

        # find classifier object (pipeline or bare)
        clf = model
        if hasattr(model, "named_steps"):
            # try common names
            for name in ("classifier", "clf", "model", "estimator"):
                if name in model.named_steps:
                    clf = model.named_steps[name]
                    break
            else:
                # fallback to last step
                clf = list(model.named_steps.values())[-1]

        # attempt to get nice label
        label = raw
        try:
            if hasattr(clf, "classes_"):
                classes = list(clf.classes_)
                # if classes are strings and raw is index-like, map
                if len(classes) > 0 and isinstance(classes[0], str):
                    # if raw is an index (0/1), try to map
                    if isinstance(raw, (np.integer, int)):
                        label = classes[int(raw)]
                    else:
                        label = raw
                else:
                    # classes are numeric (e.g., 0/1), map numeric to meaning
                    if isinstance(raw, (np.integer, int, float)):
                        label = int(raw)
                    else:
                        label = raw
        except Exception:
            label = raw

        # confidence / probability (if available)
        proba = None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                proba = float(np.max(probs[0]))
            elif hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X if clf is model else X)
                proba = float(np.max(probs[0]))
        except Exception:
            proba = None

        # if label is numeric 0/1, convert to human-friendly
        if isinstance(label, (int, np.integer)):
            label_str = "Eligible" if int(label) == 1 else "Not Eligible"
        else:
            # try to normalize common strings
            ls = str(label).strip().lower()
            if ls in ("1", "true", "yes", "eligible"):
                label_str = "Eligible"
            elif ls in ("0", "false", "no", "not eligible"):
                label_str = "Not Eligible"
            else:
                label_str = str(label)

        return label_str, proba, raw
    except Exception as e:
        st.error(f"Classification prediction failed: {e}")
        return None, None, None

# -----------------------
# UI - Header & Model selection
# -----------------------
st.title("💰 EMI Prediction & Eligibility Checker")
st.markdown(
    "Enter applicant details (or upload CSV) — app applies feature engineering and uses selected model "
    "to predict Maximum EMI (regression) and EMI Eligibility (classification)."
)
st.sidebar.header("Model & Mode")

selected_model = st.sidebar.selectbox("Choose Model", list(MODEL_CONFIG.keys()))
clf_path = MODEL_CONFIG[selected_model]["classifier"]
reg_path = MODEL_CONFIG[selected_model]["regressor"]

st.sidebar.markdown("**Files used**")
st.sidebar.text(f"Classifier: {clf_path}")
st.sidebar.text(f"Regressor: {reg_path}")

# Load models
with st.spinner("Loading models..."):
    clf_model = load_model(clf_path)
    reg_model = load_model(reg_path)

if clf_model is None or reg_model is None:
    st.warning("One or both model files not found. Please check MODEL_CONFIG paths in the script.")

# -----------------------
# Input: form or CSV upload (unchanged)
# -----------------------
mode = st.radio("Input mode", options=["Single (form)", "Batch (CSV upload)"], horizontal=True)

base_fields = {
    "age": 30,
    "gender": "Male",
    "marital_status": "Single",
    "education": "Graduate",
    "monthly_salary": 30000.0,
    "employment_type": "Salaried",
    "years_of_employment": 3.0,
    "company_type": "Private",
    "house_type": "Owned",
    "monthly_rent": 0.0,
    "family_size": 3,
    "dependents": 1,
    "school_fees": 0.0,
    "college_fees": 0.0,
    "travel_expenses": 1000.0,
    "groceries_utilities": 5000.0,
    "other_monthly_expenses": 1000.0,
    "existing_loans": "No",
    "current_emi_amount": 0.0,
    "credit_score": 700.0,
    "bank_balance": 10000.0,
    "emergency_fund": 20000.0,
    "emi_scenario": "Personal Loan",
    "requested_amount": 500000.0,
    "requested_tenure": 60
}

input_df = None

if mode == "Single (form)":
    st.subheader("Single applicant form")
    with st.form("single_form"):
        cols = st.columns(3)
        with cols[0]:
            age = st.number_input("Age", value=int(base_fields["age"]), min_value=18, max_value=100)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
            marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Widowed"])
            education = st.selectbox("Education", options=["Graduate", "Postgraduate", "High School", "PhD", "Other"], index=0)
            employment_type = st.selectbox("Employment Type", options=["Salaried", "Self-employed", "Unemployed"], index=0)
            company_type = st.selectbox("Company Type", options=["Private", "Public", "Government", "Other"], index=0)
            house_type = st.selectbox("House Type", options=["Owned", "Rented", "Family", "Other"], index=0)
            years_of_employment = st.number_input("Years of Employment", value=float(base_fields["years_of_employment"]), min_value=0.0, max_value=80.0)
        with cols[1]:
            monthly_salary = st.number_input("Monthly Salary (₹)", value=float(base_fields["monthly_salary"]), min_value=0.0)
            monthly_rent = st.number_input("Monthly Rent (₹)", value=float(base_fields["monthly_rent"]), min_value=0.0)
            family_size = st.number_input("Family Size", value=int(base_fields["family_size"]), min_value=1)
            dependents = st.number_input("Dependents", value=int(base_fields["dependents"]), min_value=0)
            current_emi_amount = st.number_input("Current EMI Amount (₹)", value=float(base_fields["current_emi_amount"]), min_value=0.0)
            existing_loans = st.selectbox("Existing Loans", options=["No", "Yes"], index=0)
            credit_score = st.number_input("Credit Score", value=float(base_fields["credit_score"]), min_value=300.0, max_value=900.0)
        with cols[2]:
            school_fees = st.number_input("School Fees (monthly)", value=float(base_fields["school_fees"]), min_value=0.0)
            college_fees = st.number_input("College Fees (monthly)", value=float(base_fields["college_fees"]), min_value=0.0)
            travel_expenses = st.number_input("Travel Expenses (monthly)", value=float(base_fields["travel_expenses"]), min_value=0.0)
            groceries_utilities = st.number_input("Groceries & Utilities (monthly)", value=float(base_fields["groceries_utilities"]), min_value=0.0)
            other_monthly_expenses = st.number_input("Other Monthly Expenses", value=float(base_fields["other_monthly_expenses"]), min_value=0.0)
            bank_balance = st.number_input("Bank Balance (₹)", value=float(base_fields["bank_balance"]), min_value=0.0)
            emergency_fund = st.number_input("Emergency Fund (₹)", value=float(base_fields["emergency_fund"]), min_value=0.0)

        st.markdown("---")
        requested_amount = st.number_input("Requested Loan Amount (₹)", value=float(base_fields["requested_amount"]), min_value=0.0)
        requested_tenure = st.number_input("Requested Tenure (months)", value=int(base_fields["requested_tenure"]), min_value=1)
        emi_scenario = st.selectbox("EMI Scenario", options=["Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Other"])
        submitted = st.form_submit_button("Run Prediction")

        if submitted:
            values = {
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
            }
            input_df = df_from_form(values)

elif mode == "Batch (CSV upload)":
    st.subheader("Upload a CSV (rows = applicants). Columns should match form fields or include at least those used for feature engineering.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded batch file with {input_df.shape[0]} rows and {input_df.shape[1]} columns.")
            with st.expander("Show uploaded CSV (first rows)"):
                st.dataframe(input_df.head(10))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            input_df = None

# -----------------------
# When we have input_df: feature engineering and predictions
# -----------------------
if input_df is not None:
    engineered = apply_feature_engineering(input_df)
    st.subheader("📊 Input Summary (engineered features)")
    with st.expander("Show engineered input (first rows)"):
        st.dataframe(engineered.head(10))

    # Prediction step
    if clf_model is None or reg_model is None:
        st.error("Model files are missing. Check configured MODEL_CONFIG paths in the script.")
    else:
        preds = []
        for idx, row in engineered.iterrows():
            row_df = pd.DataFrame([row])

            # Regression prediction (predicted maximum monthly EMI)
            reg_pred = model_predict_regression(reg_model, row_df)

            # Classification prediction (eligible / not eligible)
            cls_label, cls_proba, cls_raw = model_predict_classification(clf_model, row_df)

            # make sure requested_amount and tenure are numeric
            req_amt = float(row.get("requested_amount", 0) or 0)
            req_ten = int(row.get("requested_tenure", 0) or 0)

            # Compute approved amount when eligible:
            # approved_amount = min(requested_amount, predicted_max_emi * requested_tenure)
            approved_amount = 0.0
            approved_monthly_emi = 0.0
            approved_reason = ""
            if cls_label is not None and str(cls_label).strip().lower() == "eligible" and reg_pred is not None and req_ten > 0:
                # compute maximum loan value applicant can be given based on predicted monthly capacity
                max_possible_loan = reg_pred * req_ten
                approved_amount = min(req_amt, max_possible_loan) if req_amt > 0 else max_possible_loan
                # monthly EMI on approved amount (simple equal installment = approved_amount / tenure)
                approved_monthly_emi = approved_amount / req_ten if req_ten > 0 else 0.0
                approved_reason = f"Approved as predicted capacity supports up to ₹{max_possible_loan:,.2f} over {req_ten} months."
            else:
                approved_amount = 0.0
                approved_monthly_emi = 0.0
                if cls_label is not None:
                    approved_reason = "Not eligible or insufficient predicted EMI capacity."

            preds.append({
                **row.to_dict(),
                "predicted_max_emi": reg_pred,
                "predicted_eligibility_label": cls_label,
                "prediction_confidence": cls_proba,
                "prediction_raw_label": cls_raw,
                "approved_amount": approved_amount,
                "approved_monthly_emi": approved_monthly_emi,
                "approval_reason": approved_reason
            })

        results_df = pd.DataFrame(preds)

        st.subheader("🔎 Predictions")
        display_cols = [
            "predicted_max_emi", "predicted_eligibility_label", "prediction_confidence",
            "approved_amount", "approved_monthly_emi", "approval_reason"
        ] + [c for c in results_df.columns if c not in (
            "predicted_max_emi","predicted_eligibility_label","prediction_confidence",
            "approved_amount","approved_monthly_emi","approval_reason"
        )]
        st.dataframe(results_df[display_cols].head(100))

        # Single-record nice display
        if results_df.shape[0] == 1:
            r = results_df.iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                if pd.notnull(r["predicted_max_emi"]):
                    st.metric("📈 Predicted Maximum EMI (monthly)", f"₹ {r['predicted_max_emi']:,.2f}")
                else:
                    st.metric("📈 Predicted Maximum EMI (monthly)", "N/A")
                st.write("**Requested amount:**", f"₹ {r.get('requested_amount', 0):,.2f}")
                st.write("**Requested tenure (months):**", int(r.get('requested_tenure', 0)))
                st.write("**Predicted monthly instalment if requested amount granted:**",
                         f"₹ {(r.get('requested_amount',0) / max(1,int(r.get('requested_tenure',1)))):,.2f}")
            with c2:
                label = r["predicted_eligibility_label"]
                proba = r["prediction_confidence"]
                approved_amt = r["approved_amount"]
                approved_emi = r["approved_monthly_emi"]
                reason = r["approval_reason"]

                if label is not None:
                    if str(label).lower() == "eligible":
                        st.success(f"✅ Eligibility: {label}")
                    elif str(label).lower() in ("not eligible", "not eligible"):
                        st.error(f"🚫 Eligibility: {label}")
                    else:
                        st.info(f"ℹ️ Eligibility: {label}")

                if proba is not None:
                    try:
                        st.metric("Confidence", f"{proba:.2%}")
                    except Exception:
                        st.write("Confidence:", proba)

                # Show approved amount details
                if approved_amt and approved_amt > 0:
                    st.markdown(f"### ✅ Approved amount: ₹ {approved_amt:,.2f}")
                    st.markdown(f"**Approved monthly EMI:** ₹ {approved_emi:,.2f}")
                    st.caption(reason)
                else:
                    st.markdown("### ❌ Not approved")
                    st.caption(reason)

        # Download button for batch results
        to_download_button(results_df, filename="predictions_with_approvals.csv", label="⬇️ Download Predictions + Approvals CSV")

        st.divider()
        st.caption(
            f"Predictions done using **{selected_model}** models. "
            "Approved amount is computed as min(requested_amount, predicted_max_emi * requested_tenure). "
            "Make sure your .pkl models include preprocessing used during training (preferred) or that input columns match the training pipeline."
        )

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Built by Malay — Update model paths in MODEL_CONFIG if using different file names.")
