# xgboost_training.py

import os
import sys
import joblib
import dagshub
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mlflow

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

from data_preprocessing import load_and_preprocess_data
from upsampling import apply_smote_upsampling   # 🔥 using your upsampling file

warnings.filterwarnings("ignore")


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    dagshub.init(repo_owner="malaychand", repo_name="EMIPredict_AI", mlflow=True)
    print("✅ Connected to DagsHub MLflow server")

    git_commit = get_git_commit()
    mlflow.set_experiment("EMI_prediction_xgboost")

    # ----------------------------
    # Load dataset
    # ----------------------------
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv",
    ]
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if data_path is None:
        print("❌ Dataset not found.")
        sys.exit()

    df = load_and_preprocess_data(data_path)

    print("\n📌 Original data distribution:")
    print(df["emi_eligibility"].value_counts())

    # X, y split
    X = df.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y = df["emi_eligibility"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Label encoding
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    print("\nLabel Mapping:", dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_))))

    # Preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    # ----------------------------
    # SMOTE Upsampling
    # ----------------------------
    X_resampled, y_resampled = apply_smote_upsampling(
        X, y_encoded, preprocessor, label_encoder=label_enc
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # ----------------------------
    # XGBOOST MODEL
    # ----------------------------
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=300
    )

    # ----------------------------
    # MLflow Logging
    # ----------------------------
    with mlflow.start_run(run_name="xgboost_smote"):

        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("upsampling", "SMOTE")

        # Train XGB
        print("\n🚀 Training XGBoost...")
        xgb.fit(X_train, y_train)

        # Predictions
        y_pred = xgb.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Classification Report
        report = classification_report(
            label_enc.inverse_transform(y_test),
            label_enc.inverse_transform(y_pred)
        )
        mlflow.log_text(report, "classification_report.txt")

        # Save Model
        os.makedirs("models", exist_ok=True)
        model_path = "models/xgboost_classifier_smote.pkl"
        joblib.dump(xgb, model_path)
        mlflow.log_artifact(model_path)

        print("\n🎉 XGBoost Training Complete with SMOTE Upsampling")
        print(f"🔹 Accuracy: {acc:.4f}")
        print(f"🔹 F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
