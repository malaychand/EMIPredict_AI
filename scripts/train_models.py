"""
train_models_selected_features.py
Train classification and regression models using preprocessed features.
- Classification features: auto-selected
- Regression features: auto-selected + include emi_eligibility
- MLflow logging integrated (with signature & input example)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "artifacts" / "feature_engineered_EMI_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = MODEL_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = ROOT / "mlruns"
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI.as_posix()}")
mlflow.set_experiment("EMIPredictAI_experiment")

# -----------------------------
# Metrics
# -----------------------------
def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# -----------------------------
# Main training function
# -----------------------------
def main():
    # Load dataset
    df = pd.read_csv(FEATURE_FILE)
    print(f"✅ Loaded dataset: {df.shape}")

    # Targets
    target_clf = "emi_eligibility"
    target_reg = "max_monthly_emi"

    # -----------------------------
    # Feature selection based on correlation
    # -----------------------------
    corr_matrix = df.corr()

    clf_corr = corr_matrix[target_clf].abs()
    clf_features = clf_corr[clf_corr > 0.05].index.tolist()
    if target_clf in clf_features:
        clf_features.remove(target_clf)

    def remove_highly_correlated(features, matrix, threshold=0.85):
        selected = []
        for f in features:
            if all(abs(matrix[f][selected]) < threshold) if selected else True:
                selected.append(f)
        return selected

    clf_features = remove_highly_correlated(clf_features, corr_matrix)
    print(f"Selected classification features: {clf_features}")

    reg_corr = corr_matrix[target_reg].abs()
    reg_features = reg_corr[reg_corr > 0.05].index.tolist()
    if target_reg in reg_features:
        reg_features.remove(target_reg)

    if target_clf not in reg_features:
        reg_features.append(target_clf)

    reg_features = remove_highly_correlated(reg_features, corr_matrix)
    print(f"Selected regression features (including emi_eligibility): {reg_features}")

    # Save feature lists
    joblib.dump(clf_features, MODEL_DIR / "clf_features.joblib")
    joblib.dump(reg_features, MODEL_DIR / "reg_features.joblib")
    print("✅ Feature lists saved for prediction")

    # -----------------------------
    # Prepare datasets
    # -----------------------------
    X_clf = df[clf_features]
    y_clf = df[target_clf]

    X_reg = df[reg_features]
    y_reg = pd.to_numeric(df[target_reg], errors='coerce').to_numpy().reshape(-1, 1)

    # -----------------------------
    # Train-test split
    # -----------------------------
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )


    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )


    # -----------------------------
    # Classification models
    # -----------------------------
    classifiers = {
        "logistic": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    }

    best_clf_model = None
    best_f1 = -np.inf
    best_clf_name = ""

    for name, model in classifiers.items():
        with mlflow.start_run(run_name=f"Classifier_{name}"):
            model.fit(Xc_train, yc_train)
            y_pred = model.predict(Xc_test)
            metrics = classification_metrics(yc_test, y_pred)

            # Create input example & signature
            input_example = Xc_train.iloc[:1]
            signature = infer_signature(Xc_train, model.predict(Xc_train))

            # Log model, params, metrics
            mlflow.log_param("model_name", name)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name=f"{name}_classifier", signature=signature, input_example=input_example)

            print(f"{name} classifier metrics: {metrics}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_clf_model = model
                best_clf_name = name

    joblib.dump(best_clf_model, MODEL_DIR / "best_classifier.joblib")
    joblib.dump(metrics, METRICS_DIR / "classifier_metrics.joblib")
    print(f"✅ Best classifier saved: {best_clf_name} (F1={best_f1:.4f})")

    # -----------------------------
    # Regression models
    # -----------------------------
    regressors = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        "xgboost": XGBRegressor(random_state=42)
    }

    best_reg_model = None
    best_rmse = np.inf
    best_reg_name = ""

    for name, model in regressors.items():
        with mlflow.start_run(run_name=f"Regressor_{name}"):
            model.fit(Xr_train, yr_train.ravel())
            y_pred = model.predict(Xr_test)
            metrics = regression_metrics(yr_test.ravel(), y_pred)

            # Create input example & signature
            input_example = pd.DataFrame(Xr_train[:1], columns=reg_features)
            signature = infer_signature(Xr_train, model.predict(Xr_train))

            # Log model, params, metrics
            mlflow.log_param("model_name", name)
            mlflow.log_param("features", reg_features)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name=f"{name}_regressor", signature=signature, input_example=input_example)

            print(f"{name} regressor metrics: {metrics}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_reg_model = model
                best_reg_name = name

    joblib.dump(best_reg_model, MODEL_DIR / "best_regressor.joblib")
    joblib.dump(metrics, METRICS_DIR / "regressor_metrics.joblib")
    print(f"✅ Best regressor saved: {best_reg_name} (RMSE={best_rmse:.4f})")

    print("\n🎯 Training complete with selected features (MLflow logging enabled)")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
