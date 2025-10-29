# streamlit_app/page/analysis.py
import streamlit as st
from pathlib import Path
from PIL import Image

def show_analysis(REPORTS_DIR: Path):
    st.header("📊 Exploratory Data Analysis")
    st.write("Explore the dataset distributions, correlations, and financial insights.")

    # ✅ Gather all images
    image_files = sorted(
        [f for f in REPORTS_DIR.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if not image_files:
        st.warning("⚠️ No images found in the reports/ folder. Please run `generate_plots.py` first.")
        return

    st.subheader("📈 Generated Visual Reports")

    # Display images 2 per row
    for i in range(0, len(image_files), 2):
        cols = st.columns(2)
        for j, img_path in enumerate(image_files[i:i+2]):
            with cols[j]:
                try:
                    st.image(str(img_path), width=400)  # Fixed width to avoid errors
                except Exception as e:
                    st.error(f"⚠️ Could not display image {img_path.name}: {e}")
