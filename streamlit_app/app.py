# streamlit_app/app.py

import streamlit as st
from pathlib import Path
import sys

# -----------------------------
# Add project root + scripts/ to sys.path
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))

# -----------------------------
# Import predict function from scripts
# -----------------------------
from predict_emi import predict_emi

# -----------------------------
# Page Imports
# -----------------------------
from page.home import show_home
from page.prediction import show_prediction
from page.analysis import show_analysis
from page.model_performance import show_model_performance
from page.admin import show_admin
from page.recommendations import show_recommendations

# -----------------------------
# Configurations
# -----------------------------
st.set_page_config(page_title="💰 EMIPredict AI", layout="wide")

MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", (
    "🏠 Home",
    "🔍 Prediction",
    "📊 Analysis",
    "📈 Model Performance",
    "🛠️ Admin",
    "💡 Recommendations"
))
# -----------------------------
# Render Selected Page
# -----------------------------
if page == "🏠 Home":
    show_home()
elif page == "🔍 Prediction":
    show_prediction(MODEL_DIR)
elif page == "📊 Analysis":
    show_analysis(REPORTS_DIR)
elif page == "📈 Model Performance":
    show_model_performance(MODEL_DIR)
elif page == "🛠️ Admin":
    show_admin(ROOT)
elif page == "💡 Recommendations":
    show_recommendations()
else:
    st.write("Page not found")

