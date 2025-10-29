import streamlit as st
from pathlib import Path
import pandas as pd

def show_admin(ROOT):
    st.title("🛠️ Admin Panel")
    st.markdown("""
    Admin tools for dataset management and operations.
    """)

    ARTIFACTS_DIR = ROOT / "artifacts"
    CLEAN_FILE = ARTIFACTS_DIR / "cleaned_EMI_dataset.csv"

    if CLEAN_FILE.exists():
        st.subheader("Current Dataset Preview")
        df = pd.read_csv(CLEAN_FILE)
        st.dataframe(df.head(10))
        st.write(f"Total Records: {len(df)}")

        st.subheader("Download Dataset")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cleaned_EMI_dataset.csv",
            mime='text/csv'
        )
    else:
        st.warning("Dataset not found in `artifacts/` folder.")
