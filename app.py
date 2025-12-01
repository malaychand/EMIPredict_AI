# app.py (multi-page single-file)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="EMIPredict AI", layout="wide")

ROOT = Path.cwd()
CLEAN_FILE = ROOT / "data" / "cleaned_EMI_dataset.csv"
MODEL_DIR = ROOT / "models"

MODEL_CONFIG = {
    "XGBoost": {"classifier": "xgb_classifier.pkl", "regressor": "xgb_regressor.pkl"},
    "Logistic/Linear Regression": {"classifier": "logistic_classifier.pkl", "regressor": "linear_regressor.pkl"},
}

# -------------------------
# Project metadata (Home)
# -------------------------
PROJECT_META = {
    "title": "EMIPredict AI - Intelligent Financial Risk Assessment Platform",
    "skills": [
        "Python", "Streamlit Cloud Deployment", "Machine Learning", "Data Analysis",
        "MLflow", "Classification Models", "Regression Models", "Feature Engineering", "Data Preprocessing"
    ],
    "domain": "FinTech and Banking",
    "problem_statement": (
        "Build a comprehensive financial risk assessment platform that integrates machine "
        "learning models with MLflow experiment tracking to create an interactive web "
        "application for EMI prediction. The platform provides classification (EMI eligibility) "
        "and regression (maximum EMI amount) predictions using a large, realistic dataset."
    ),
    "dataset_summary": (
        "Dataset: EMI_dataset | Total records: 400,000 | Input features: 22 | Targets: emi_eligibility (classification) "
        "and max_monthly_emi (regression) | 5 EMI scenarios (Personal, Vehicle, Education, Home Appliances, E-commerce)."
    )
}

# -------------------------
# Utility: load dataset
# -------------------------
@st.cache_data
def load_data(path: Path):
    return pd.read_csv(path)

# -------------------------
# Utility: build preprocessor
# -------------------------
def build_preprocessor(numerical_cols, categorical_cols):
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])
    return preprocessor

# -------------------------
# Navigation
# -------------------------
PAGES = ["Home", "Prediction", "Report"]
page = st.sidebar.radio("Navigation", PAGES)

# Put model selection in sidebar (used by Prediction page)
st.sidebar.markdown("---")
selected_model = st.sidebar.selectbox("Choose ML Algorithm", options=list(MODEL_CONFIG.keys()))

# Try load dataset (used in multiple pages)
try:
    df = load_data(CLEAN_FILE)
except Exception as e:
    if page == "Home":
        # show Home page but warn if dataset missing
        st.sidebar.error(f"Dataset not found: {e}")
    else:
        st.error(f"Failed to load dataset at {CLEAN_FILE}. Error: {e}")
        st.stop()
else:
    # proceed to infer columns only when dataset loaded
    TARGET_CLF = "emi_eligibility"
    TARGET_REG = "max_monthly_emi"
    X = df.drop(columns=[TARGET_CLF, TARGET_REG])
    y_clf = df[TARGET_CLF]
    y_reg = df[TARGET_REG]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # preprocessor and label encoder (fit to dataset for consistent ordering)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    try:
        with st.spinner("Fitting preprocessor..."):
            preprocessor.fit(X)
    except Exception as e:
        st.error(f"Preprocessor fitting failed: {e}")
        st.stop()

    label_enc = LabelEncoder()
    label_enc.fit(y_clf)

# -------------------------
# Home Page
# -------------------------
def render_home():
    # Title + Subtitle
    st.title(f"🚀 {PROJECT_META['title']}")
    st.caption(f"🌐 Domain: **{PROJECT_META['domain']}**")

    st.markdown("---")

    # Problem statement
    st.markdown("## ❓ Problem Statement")
    st.write(PROJECT_META["problem_statement"])

    # MLflow Tracking Link
    st.markdown("## 🧪 MLflow Experiment Tracking Dashboard")
    st.write("All model experiments, metrics, artifacts and versioned models are logged here:")

    st.markdown(
        """
        <a href="https://dagshub.com/malaychand/EMIPredict_AI.mlflow" target="_blank">
            <button style="background-color:#ff4b4b; color:white; padding:10px 25px; 
            border-radius:8px; border:none; font-size:16px; cursor:pointer; margin-bottom:10px;">
                🔍 Open MLflow Dashboard
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

    # GitHub Repo Link
    st.markdown("## 🧭 Project Repository (GitHub)")
    st.write("Explore complete source code, models, notebooks, and deployment files:")

    st.markdown(
        """
        <a href="https://github.com/malaychand/EMIPredict_AI" target="_blank">
            <button style="background-color:#4b9ce2; color:white; padding:10px 25px; 
            border-radius:8px; border:none; font-size:16px; cursor:pointer;">
                💻 View GitHub Repository
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Business Use Cases section in two columns
    st.markdown("## 💼 Business Use Cases")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏦 Financial Institutions")
        st.write("""
        - Automate loan approvals  
        - Risk-based pricing  
        - Real-time customer eligibility  
        """)

        st.markdown("### 📱 FinTech Platforms")
        st.write("""
        - Instant EMI eligibility checks  
        - Mobile banking integration  
        - Automated risk scoring  
        """)

    with col2:
        st.markdown("### 🏛 Banks & Credit Agencies")
        st.write("""
        - Portfolio risk management  
        - Data-driven loan limits  
        - Regulatory-ready documentation  
        """)

        st.markdown("### 👩‍💼 Loan Officers")
        st.write("""
        - AI recommendations  
        - Full applicant profile in seconds  
        - Track historical performance  
        """)

    st.markdown("---")

    # Approach
    st.markdown("## 🛠 Project Approach (Step-by-Step)")
    st.write("""
    1️⃣ **Data Cleaning & Preprocessing** (feature engineering, quality checks)  
    2️⃣ **Exploratory Data Analysis** (correlations, distributions, demographic insights)  
    3️⃣ **Model Training** (Logistic, RF, XGBoost — both classification & regression)  
    4️⃣ **MLflow Experiment Tracking** (metrics, artifacts, version control)  
    5️⃣ **Streamlit App Development** (model inference UI)  
    6️⃣ **Streamlit Cloud Deployment** (production-ready)  
    """)

    st.markdown("---")

    st.success("🎯 Use the left navigation sidebar to access **Prediction** or **Report** sections.")

# -------------------------
# Prediction Page
# -------------------------
def render_prediction():
    st.header("💸 EMI Eligibility & Max-Monthly-EMI Predictor")

    # Show model info
    st.markdown(f"**Selected model:** {selected_model}")
    classifier_path = MODEL_DIR / MODEL_CONFIG[selected_model]["classifier"]
    regressor_path = MODEL_DIR / MODEL_CONFIG[selected_model]["regressor"]

    # Show basic dataset info and head/describe
    with st.expander("Dataset preview (head & describe)"):
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.dataframe(df.head(6))

    loaded_models = {}
    if classifier_path.exists():
        try:
            loaded_models["classifier"] = joblib.load(classifier_path)
            st.success(f"Loaded classifier: {classifier_path.name}")
        except Exception as e:
            st.error(f"Failed to load classifier: {e}")
    else:
        st.warning(f"Classifier not found at {classifier_path}")

    if regressor_path.exists():
        try:
            loaded_models["regressor"] = joblib.load(regressor_path)
            st.success(f"Loaded regressor: {regressor_path.name}")
        except Exception as e:
            st.error(f"Failed to load regressor: {e}")
    else:
        st.warning(f"Regressor not found at {regressor_path}")

    st.markdown("---")
    st.subheader("Enter applicant financial details (single row)")

    # Build unique values & defaults
    unique_vals = {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols}
    defaults = {
        "age": 30, "gender": unique_vals.get("gender", ["Male"])[0],
        "marital_status": unique_vals.get("marital_status", ["Single"])[0],
        "education": unique_vals.get("education", ["Graduate"])[0],
        "monthly_salary": float(df["monthly_salary"].median()),
        "employment_type": unique_vals.get("employment_type", ["Private"])[0],
        "years_of_employment": float(df["years_of_employment"].median()),
        "company_type": unique_vals.get("company_type", ["Private"])[0],
        "house_type": unique_vals.get("house_type", ["Own"])[0],
        "monthly_rent": float(df["monthly_rent"].median()),
        "family_size": int(df["family_size"].median()),
        "dependents": int(df["dependents"].median()),
        "school_fees": float(df["school_fees"].median()),
        "college_fees": float(df["college_fees"].median()),
        "travel_expenses": float(df["travel_expenses"].median()),
        "groceries_utilities": float(df["groceries_utilities"].median()),
        "other_monthly_expenses": float(df["other_monthly_expenses"].median()),
        "existing_loans": unique_vals.get("existing_loans", ["No"])[0],
        "current_emi_amount": float(df["current_emi_amount"].median()),
        "credit_score": float(df["credit_score"].median()),
        "bank_balance": float(df["bank_balance"].median()),
        "emergency_fund": float(df["emergency_fund"].median()),
        "emi_scenario": unique_vals.get("emi_scenario", ["Personal Loan EMI"])[0],
        "requested_amount": float(df["requested_amount"].median()),
        "requested_tenure": float(df["requested_tenure"].median())
    }

    # Form
    with st.form("single_row_form"):
        cols = st.columns(3)
        with cols[0]:
            age = st.number_input("Age", value=int(defaults["age"]), min_value=18, max_value=100)
            gender = st.selectbox("Gender", options=unique_vals.get("gender", ["Male", "Female"]), index=0)
            marital_status = st.selectbox("Marital Status", options=unique_vals.get("marital_status", ["Single", "Married"]), index=0)
            education = st.selectbox("Education", options=unique_vals.get("education", ["Graduate", "Post Graduate", "High School"]), index=0)
            employment_type = st.selectbox("Employment Type", options=unique_vals.get("employment_type", ["Private", "Government", "Self-Employed"]), index=0)
            company_type = st.selectbox("Company Type", options=unique_vals.get("company_type", ["Mid-Size", "Mnc", "Startup", "Large Indian", "Small"]), index=0)
            house_type = st.selectbox("House Type", options=unique_vals.get("house_type", ["Rented", "Family", "Own"]), index=0)
            years_of_employment = st.number_input("Years of Employment", value=float(defaults["years_of_employment"]), min_value=0.0, max_value=80.0)
        with cols[1]:
            monthly_salary = st.number_input("Monthly Salary (₹)", value=float(defaults["monthly_salary"]), min_value=0.0)
            monthly_rent = st.number_input("Monthly Rent (₹)", value=float(defaults["monthly_rent"]), min_value=0.0)
            family_size = st.number_input("Family Size", value=int(defaults["family_size"]), min_value=1)
            dependents = st.number_input("Dependents", value=int(defaults["dependents"]), min_value=0)
            current_emi_amount = st.number_input("Current EMI Amount (₹)", value=float(defaults["current_emi_amount"]), min_value=0.0)
            existing_loans = st.selectbox("Existing Loans", options=unique_vals.get("existing_loans", ["No", "Yes"]), index=0)
            credit_score = st.number_input("Credit Score", value=float(defaults["credit_score"]), min_value=300.0, max_value=900.0)
        with cols[2]:
            school_fees = st.number_input("School Fees (monthly)", value=float(defaults["school_fees"]), min_value=0.0)
            college_fees = st.number_input("College Fees (monthly)", value=float(defaults["college_fees"]), min_value=0.0)
            travel_expenses = st.number_input("Travel Expenses (monthly)", value=float(defaults["travel_expenses"]), min_value=0.0)
            groceries_utilities = st.number_input("Groceries & Utilities (monthly)", value=float(defaults["groceries_utilities"]), min_value=0.0)
            other_monthly_expenses = st.number_input("Other Monthly Expenses", value=float(defaults["other_monthly_expenses"]), min_value=0.0)
            bank_balance = st.number_input("Bank Balance (₹)", value=float(defaults["bank_balance"]), min_value=0.0)
            emergency_fund = st.number_input("Emergency Fund (₹)", value=float(defaults["emergency_fund"]), min_value=0.0)
        st.markdown("---")
        emi_scenario = st.selectbox("EMI Scenario", options=unique_vals.get("emi_scenario", ["Personal Loan EMI", "Vehicle EMI", "Education EMI"]), index=0)
        requested_amount = st.number_input("Requested Loan Amount (₹)", value=float(defaults["requested_amount"]), min_value=0.0)
        requested_tenure = st.number_input("Requested Tenure (months)", value=float(defaults["requested_tenure"]), min_value=1.0)
        submitted = st.form_submit_button("Predict (classification + regression)")

    def build_input_df():
        row = {
            "age": int(age),
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": float(monthly_salary),
            "employment_type": employment_type,
            "years_of_employment": float(years_of_employment),
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": float(monthly_rent),
            "family_size": int(family_size),
            "dependents": int(dependents),
            "school_fees": float(school_fees),
            "college_fees": float(college_fees),
            "travel_expenses": float(travel_expenses),
            "groceries_utilities": float(groceries_utilities),
            "other_monthly_expenses": float(other_monthly_expenses),
            "existing_loans": existing_loans,
            "current_emi_amount": float(current_emi_amount),
            "credit_score": float(credit_score),
            "bank_balance": float(bank_balance),
            "emergency_fund": float(emergency_fund),
            "emi_scenario": emi_scenario,
            "requested_amount": float(requested_amount),
            "requested_tenure": float(requested_tenure)
        }
        input_df = pd.DataFrame([row])
        input_df = input_df.reindex(columns=X.columns)
        return input_df

    if submitted:
        input_df = build_input_df()
        st.subheader("Input row (as DataFrame)")
        st.dataframe(input_df)

        # Classifier prediction
        if "classifier" in loaded_models:
            clf = loaded_models["classifier"]
            try:
                X_in_clf = preprocessor.transform(input_df)
                pred_num = clf.predict(X_in_clf)
                probs = clf.predict_proba(X_in_clf)[0] if hasattr(clf, "predict_proba") else None
                pred_label = label_enc.inverse_transform(pred_num.astype(int))[0]
                st.markdown("### Classification (Eligibility)")
                st.write(f"**Predicted label:** {pred_label}")
                if probs is not None:
                    classes = label_enc.inverse_transform(np.arange(len(label_enc.classes_)))
                    prob_df = pd.DataFrame({"class": classes, "probability": probs})
                    prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
                    st.dataframe(prob_df.style.format({"probability": "{:.3f}"}))
            except Exception as e:
                st.error(f"Failed to run classifier prediction: {e}")
        else:
            st.warning("Classifier model not available. Place the classifier .pkl in the models/ folder.")

        # Regressor prediction
        if "regressor" in loaded_models:
            reg = loaded_models["regressor"]
            try:
                y_pred_reg = reg.predict(input_df)
                st.markdown("### Regression (Predicted max_monthly_emi)")
                st.write(f"**Predicted max_monthly_emi:** ₹ {float(y_pred_reg[0]):,.2f}")
            except Exception as e:
                st.error(f"Failed to run regressor prediction: {e}")
        else:
            st.warning("Regressor model not available. Place the regressor .pkl in the models/ folder.")

# -------------------------
# Report Page
# -------------------------
def render_report():
    st.header("📊 Report & Data Exploration")

    # -----------------------------
    # Dataset preview inside an expander
    # -----------------------------
    with st.expander("📁 Dataset preview (head & describe)"):
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.dataframe(df.head(10))

        st.markdown("### Summary statistics for numeric columns")
        st.dataframe(df.describe().T.style.format("{:.2f}"))

        st.markdown("### Column information")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Unique Values": df.nunique().values
        })
        st.dataframe(col_info)

    st.markdown("---")

    # -----------------------------
    # Automatically load all generated EDA images
    # -----------------------------
    st.subheader("📈 Exploratory Data Analysis (Image Gallery)")

    REPORT_DIR = ROOT / "data" / "reports"
    images = sorted(REPORT_DIR.glob("*.png"))  # all images produced by your script

    if not images:
        st.info("No EDA images found in data/reports/")
        return

    st.markdown("#### All EDA Images (Displayed Side-by-Side)")

    # A mapping to show short description depending on image name
    description_map = {
        "emi_eligibility_distribution": "Shows the distribution of EMI eligibility classes.",
        "numeric_features_histograms": "Histogram overview of all numeric financial variables.",
        "affordability_ratio_distribution": "Distribution of affordability ratio across applicants.",
        "affordability_ratio_vs": "Relation between affordability ratio and financial variables.",
        "debt_to_income_vs": "Scatter plot showing how debt-to-income relates to major features.",
        "expense_to_income_vs": "Scatter plot showing how expense-to-income relates to major features.",
    }

    # Display images in rows of 2 (side-by-side)
    cols_per_row = 2
    rows = (len(images) + cols_per_row - 1) // cols_per_row

    for r in range(rows):
        cols = st.columns(cols_per_row)

        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(images):
                img_path = images[idx]
                name = img_path.stem  # filename without extension

                # Determine short description from lookup
                description = "Insightful visualization useful for financial risk analysis."
                for key in description_map:
                    if key in name:
                        description = description_map[key]
                        break

                with cols[c]:
                    st.image(str(img_path), caption=name.replace("_", " ").title(), use_container_width=True)

                    # one-line description
                    st.write(f"📝 **{description}**")

    st.markdown("---")

    st.success("📄 All EDA visualizations loaded successfully.")



# -------------------------
# Page router
# -------------------------
if page == "Home":
    render_home()
elif page == "Prediction":
    render_prediction()
elif page == "Report":
    render_report()
else:
    st.error("Unknown page")
