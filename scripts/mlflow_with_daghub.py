# linear_regression_baseline.py
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from data_preprocessing import load_and_preprocess_data
warnings.filterwarnings('ignore')


# ----------------------------
# Helper: Get Git Commit Hash
# ----------------------------
def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


# ----------------------------
# Main Training Function
# ----------------------------
def main():
    # ✅ Connect to DagsHub for MLflow Tracking
    dagshub.init(repo_owner='malaychand', repo_name='EMIPredict_AI', mlflow=True)
    print("✅ Connected to DagsHub MLflow Tracking Server")

    # ✅ Locate dataset
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv"
    ]
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if data_path is None:
        sys.exit("❌ Dataset not found. Please verify path.")

    # ✅ Load and preprocess
    df = load_and_preprocess_data(data_path)
    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['max_monthly_emi']

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Preprocessing pipeline
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # ✅ Model pipeline (Linear Regression)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # ✅ Define hyperparameter space
    param_distributions = {
        'regressor__fit_intercept': [True, False],
        'regressor': [LinearRegression(), Ridge(), Lasso()],
        'regressor__positive': [True, False]
    }

    # ✅ Randomized Search CV (5 iterations)
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=2,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    git_commit = get_git_commit()
    mlflow.set_experiment("emi_prediction_regression")

    with mlflow.start_run(run_name="linear_regression_randomsearch") as run:
        # ✅ Fit Randomized Search
        random_search.fit(X_train, y_train)

        # ✅ Extract best model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        # ✅ Predictions
        y_pred = best_model.predict(X_test)

        # ✅ Evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # ✅ Log hyperparameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        })

        # ✅ Visualization: Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual EMI")
        plt.ylabel("Predicted EMI")
        plt.title("Actual vs Predicted EMI (Linear Regression with Random Search)")
        plt.tight_layout()
        act_pred_path = "linear_regression_actual_vs_pred.png"
        plt.savefig(act_pred_path, dpi=150)
        plt.close()
        mlflow.log_artifact(act_pred_path)
        os.remove(act_pred_path)

        # ✅ Residual Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.4)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted EMI")
        plt.ylabel("Residuals")
        plt.title("Residual Plot (Linear Regression)")
        plt.tight_layout()
        residuals_path = "linear_regression_residuals.png"
        plt.savefig(residuals_path, dpi=150)
        plt.close()
        mlflow.log_artifact(residuals_path)
        os.remove(residuals_path)

        # ✅ Save and Log Model (DagsHub-compatible)
        joblib.dump(best_model, "linear_regression_model.pkl")
        mlflow.log_artifact("linear_regression_model.pkl")

        # ✅ Log model signature manually
        signature = infer_signature(X_train, best_model.predict(X_train[:5]))
        signature_path = "linear_regression_signature.txt"
        with open(signature_path, "w") as f:
            f.write(str(signature))
        mlflow.log_artifact(signature_path)
        os.remove(signature_path)

        # ✅ Metadata
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        mlflow.log_artifact(__file__)

        print(f"\n✅ Linear Regression (RandomizedSearchCV) Completed")
        print(f"🏆 Best Params: {best_params}")
        print(f"📈 RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}")
        print(f"🔗 Run URL: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
