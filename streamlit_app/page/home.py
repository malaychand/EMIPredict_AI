# streamlit_app/page/home.py
import streamlit as st
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_home():
    st.title("🏠 EMIPredict AI")
    st.markdown("""
    Welcome to **EMIPredict AI**, your intelligent financial assistant for EMI planning.  
    Plan your finances smartly with predictive insights on EMI eligibility and affordability.
    """)

    # -----------------------------
    # Key Features Section
    # -----------------------------
    st.markdown("### ✨ Key Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 💳 EMI Prediction")
        st.markdown("Predict EMI Eligibility and Maximum Monthly EMI accurately using AI models.")

    with col2:
        st.markdown("#### 📚 Data Analysis")
        st.markdown("Visualize dataset distributions, correlations, and trends interactively.")

    with col3:
        st.markdown("#### 📈 Model Performance")
        st.markdown("Monitor accuracy, F1 scores, RMSE, and other metrics of ML models.")

    col4, col5,col6 = st.columns(3)
    with col4:
        st.markdown("#### 🛠 Recommendations")
        st.markdown("Get actionable insights for improving EMI eligibility and financial health.")

    with col5:
        st.markdown("#### ⚙️ Admin Tools")
        st.markdown("Manage datasets, features, and model retraining efficiently.")

    st.markdown("---")

    
