# streamlit_app/page/model_performance.py

import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def show_model_performance(MODEL_DIR):
    st.title("📈 Model Performance")

    # -----------------------------
    # Load Best Models
    # -----------------------------
    clf_file = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_classifier")][0]
    reg_file = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_regressor")][0]

    clf = joblib.load(Path(MODEL_DIR) / clf_file)
    reg = joblib.load(Path(MODEL_DIR) / reg_file)

    # -----------------------------
    # Load Metrics (if saved separately)
    # -----------------------------
    metrics_dir = Path(MODEL_DIR) / "metrics"
    clf_metrics_file = metrics_dir / "classifier_metrics.joblib"
    reg_metrics_file = metrics_dir / "regressor_metrics.joblib"

    if clf_metrics_file.exists():
        clf_metrics = joblib.load(clf_metrics_file)
    else:
        clf_metrics = {"accuracy": "N/A", "f1": "N/A", "precision": "N/A", "recall": "N/A", "roc_auc": "N/A"}

    if reg_metrics_file.exists():
        reg_metrics = joblib.load(reg_metrics_file)
    else:
        reg_metrics = {"rmse": "N/A", "mae": "N/A", "r2": "N/A", "mape": "N/A"}

    # -----------------------------
    # Display Best Models
    # -----------------------------
    st.subheader("✅ Best Classifier")
    st.write(f"**Model:** {clf.__class__.__name__}")
    st.write("**Metrics:**")
    st.json(clf_metrics)

    st.subheader("✅ Best Regressor")
    st.write(f"**Model:** {reg.__class__.__name__}")
    st.write("**Metrics:**")
    st.json(reg_metrics)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("📊 Feature Importance")

    def plot_feature_importance(model, title="Feature Importance"):
        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_)
            else:
                st.info("Feature importance not available for this model.")
                return

            if hasattr(model, "feature_names_in_"):
                features = model.feature_names_in_
            else:
                features = [f"Feature {i}" for i in range(len(importance))]

            fi_df = pd.DataFrame({"feature": features, "importance": importance})
            fi_df = fi_df.sort_values(by="importance", ascending=False).head(20)

            plt.figure(figsize=(10,6))
            sns.barplot(x="importance", y="feature", data=fi_df, palette="viridis")
            plt.title(title)
            st.pyplot(plt)
        except Exception as e:
            st.warning(f"Could not plot feature importance: {e}")

    plot_feature_importance(clf, title="Classifier Feature Importance")
    plot_feature_importance(reg, title="Regressor Feature Importance")

    # -----------------------------
    # MLflow Dashboard Link
    # -----------------------------
    # -----------------------------
    # MLflow Dashboard Link
    # -----------------------------
    # -----------------------------
    # MLflow Dashboard Link
    # -----------------------------
    import subprocess
    import platform

    st.subheader("🔗 MLflow Dashboard")
    mlruns_path = Path(MODEL_DIR).parent / "mlruns"

    if mlruns_path.exists():
        mlflow_ui_command = f"mlflow ui --backend-store-uri file:///{mlruns_path.as_posix()}"
        st.markdown(
            f"""
            **Check your MLflow runs here:**  
            📁 `{mlruns_path}`  
            *(Open locally or via MLflow UI)*  

            To launch the MLflow dashboard manually, run:
            ```bash
            {mlflow_ui_command}
            ```
            Then visit 👉 [http://localhost:5000](http://localhost:5000)
            """,
            unsafe_allow_html=True
        )

        if st.button("🚀 Open MLflow Dashboard"):
            try:
                # Detect OS and launch MLflow UI
                if platform.system() == "Windows":
                    subprocess.Popen(["cmd", "/c", mlflow_ui_command], creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen(["bash", "-c", mlflow_ui_command])
                st.success("MLflow Dashboard is launching... Visit http://localhost:5000 in your browser.")
            except Exception as e:
                st.error(f"❌ Failed to launch MLflow UI: {e}")

    else:
        st.warning("MLflow folder not found. Make sure you have run training with MLflow tracking enabled.")
