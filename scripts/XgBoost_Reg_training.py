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
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor
from data_preprocessing import load_and_preprocess_data

warnings.filterwarnings("ignore")


# ----------------------------
# Helper: Get Git Commit Hash
# ----------------------------
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def main():
    # ✅ Connect MLflow (DagsHub)
    dagshub.init(repo_owner="malaychand", repo_name="EMIPredict_AI", mlflow=True)
    print("✅ Connected to DagsHub MLflow Tracking Server")

    git_commit = get_git_commit()
    mlflow.set_experiment("EMI_prediction_regression")

    # ==============================
    # Locate dataset
    # ==============================
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv",
    ]

    data_path = next((path for path in possible_paths if os.path.exists(path)), None)
    if data_path is None:
        sys.exit("❌ Dataset not found. Please verify path.")

    # ==============================
    # Load and preprocess
    # ==============================
    df = load_and_preprocess_data(data_path)

    X = df.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y = df["max_monthly_emi"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"📊 Data split: Train={X_train.shape}, Test={X_test.shape}")

    # ==============================
    # Preprocessing
    # ==============================
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # ==============================
    # XGBoost Regressor
    # ==============================
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        eval_metric="rmse",
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", xgb)])

    # Hyperparameter grid
    param_dist = {
        "regressor__n_estimators": [100, 200, 300, 400],
        "regressor__max_depth": [3, 5, 7, 9],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__gamma": [0, 0.1, 0.3],
        "regressor__reg_alpha": [0, 0.5, 1.0],
        "regressor__reg_lambda": [1, 1.5, 2.0],
    }

    # ==============================
    # MLflow Parent Run
    # ==============================
    with mlflow.start_run(run_name="xgboost_randomsearch_parent") as parent_run:
        print("\n🔍 Starting RandomizedSearchCV...")
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=10,
            scoring="r2",
            verbose=2,
            cv=3,
            random_state=42,
            n_jobs=-1,
            return_train_score=True,
        )
        random_search.fit(X_train, y_train)

        cv_results = pd.DataFrame(random_search.cv_results_)
        print(f"\n✅ Total iterations: {len(cv_results)}")

        # ✅ Log all RandomizedSearchCV iterations as child runs
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"iteration_{idx+1}", nested=True) as _:
                params = {
                    k.replace("param_", ""): cv_results.loc[idx, k]
                    for k in cv_results.columns
                    if k.startswith("param_")
                }
                mlflow.log_params(params)
                for fold in range(3):
                    mlflow.log_metric(f"split{fold}_test_score", cv_results.loc[idx, f"split{fold}_test_score"])

        # ✅ Best model & evaluation
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metrics(
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape,
            }
        )

        # ==============================
        # Visualizations (logged directly to MLflow)
        # ==============================
        # 1️⃣ Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        plt.xlabel("Actual EMI")
        plt.ylabel("Predicted EMI")
        plt.title("Actual vs Predicted EMI (XGBoost)")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "plots/xgb_actual_vs_pred.png")
        plt.close()

        # 2️⃣ Residual Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.4)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted EMI")
        plt.ylabel("Residuals")
        plt.title("Residual Plot (XGBoost)")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "plots/xgb_residuals.png")
        plt.close()

        # 3️⃣ Feature Importance
        feature_names = (
            numerical_cols
            + list(
                best_model.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(categorical_cols)
            )
        )
        importances = best_model.named_steps["regressor"].feature_importances_
        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
            "Importance", ascending=False
        )[:20]

        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=feat_imp)
        plt.title("Top 20 Feature Importances (XGBoost)")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "plots/xgb_feature_importance.png")
        plt.close()

        # 4️⃣ Save Model to models/ and log to MLflow
        os.makedirs("models", exist_ok=True)
        model_path = "models/xgb_best_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        # ✅ Log model signature
        signature = infer_signature(X_train, best_model.predict(X_train[:5]))
        mlflow.log_text(str(signature), "xgb_model_signature.txt")

        # ==============================
        # Metadata & Summary
        # ==============================
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("model", "XGBoostRegressor")
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")

        print("\n✅ XGBoost Regression (RandomizedSearchCV) Completed")
        print(f"🏆 Best Params: {best_params}")
        print(f"📊 Test Metrics:")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MAE: {mae:.2f}")
        print(f"   - R²: {r2:.4f}")
        print(f"   - MAPE: {mape:.2f}")
        print(f"\n🔗 MLflow URL: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()