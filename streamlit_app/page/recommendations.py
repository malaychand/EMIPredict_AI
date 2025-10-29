# streamlit_app/page/recommendations.py
import streamlit as st

def show_recommendations(clf_pred=None, reg_pred=None):
    st.title("💡 Financial Recommendations")

    # General tips (always visible)
    st.subheader("📌 General Recommendations for Applicants")
    st.markdown("""
    - Keep **Debt-to-Income ratio** below 40%
    - Maintain **Expense-to-Income ratio** below 50%
    - Ensure **affordability ratio > 1** before applying for loans
    - Build an **emergency fund** for financial stability
    - Improve **credit score** to get better EMI eligibility
    """)

    st.subheader("⚠️ Tips for Reducing Risk")
    st.markdown("""
    - Avoid multiple high EMIs simultaneously
    - Track monthly expenses and reduce unnecessary spending
    - Increase savings and emergency fund
    - Choose **loan tenure wisely** to keep EMIs manageable
    - Regularly monitor **credit score**
    """)

    # Personalized recommendations based on prediction
    if clf_pred is not None and reg_pred is not None:
        st.subheader("📝 Personalized Recommendations")
        
        if clf_pred == "Eligible":
            st.success(f"✅ You are eligible for EMI. Estimated maximum EMI: ₹ {reg_pred:,.2f}")
            st.markdown("""
            - Try to **borrow within your maximum EMI** to avoid financial strain
            - Keep **existing EMIs low** to maintain affordability
            - Consider increasing loan tenure to **reduce monthly EMI** if needed
            """)

        elif clf_pred == "High_Risk":
            st.warning(f"🟠 High Risk! Estimated maximum EMI: ₹ {reg_pred:,.2f}")
            st.markdown("""
            - Your financial profile indicates **high risk** for new loans
            - Reduce **current liabilities** and outstanding EMIs
            - Increase **savings and emergency fund**
            - Improve **credit score** before applying for a new loan
            - Consider lowering requested loan amount or extending tenure
            """)

        elif clf_pred == "Not_Eligible":
            st.error(f"🔴 Not Eligible! Estimated maximum EMI: ₹ {reg_pred:,.2f}")
            st.markdown("""
            - Currently **not eligible** for EMI
            - Reduce **existing liabilities** first
            - Increase **monthly savings and emergency fund**
            - Improve **credit score**
            - Reassess requested loan amount or tenure to meet affordability
            """)

        else:
            st.info("ℹ️ No prediction available. Please check the Prediction page first.")
