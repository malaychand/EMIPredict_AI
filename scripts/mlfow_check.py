# linear_regression_tuning.py

import os
import sys
import warnings
import joblib
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from data_preprocessing import load_and_preprocess_data
warnings.filterwarnings('ignore')


# ----------------------------
# 🔧 Helper: Get Git Commit Hash
# ----------------------------
def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown (not a git repo or git not found)"


def main():
    # ==============================
    # Locate dataset
    # ==============================
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv"
    ]
    data_path = next((path for path in possible_paths if os.path.exists(path)), None)
    if data_path is None:
        print("❌ Dataset not found. Please provide path:")
        data_path = input("Dataset path: ").strip()
        if not os.path.exists(data_path):
            sys.exit(f"❌ File not found at {data_path}")

    # ==============================
    # Load and preprocess
    # ==============================
    df = load_and_preprocess_data(data_path)
    
    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['max_monthly_emi']

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n📊 Features: {len(numerical_cols)} numeric, {len(categorical_cols)} categorical")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🧩 Split: Train={X_train.shape}, Test={X_test.shape}")

    # ==============================
    # Preprocessing pipeline
    # ==============================
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(include_bias=False))
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # ==============================
    # Model setup
    # ==============================
    model = Ridge(random_state=42)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # ==============================
    # Hyperparameter tuning
    # ==============================
    param_dist = {
        'preprocessor__num__poly__degree': [1, 2, 3],
        'regressor': [Ridge(), Lasso(), ElasticNet()],
        'regressor__alpha': np.linspace(0.001, 5, 20)
    }

    git_commit = get_git_commit()

    # ----------------------------
    # 🚀 MLflow Setup
    # ----------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("emi_prediction_regression")

    with mlflow.start_run(run_name="linear_regression_random_search") as parent_run:
        print("\n🔍 Starting RandomizedSearchCV (3-fold CV, 25 iterations)...")
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=3,
            scoring='neg_root_mean_squared_error',
            verbose=1,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )

        random_search.fit(X_train, y_train)
        results = random_search.cv_results_

        # ----------------------------
        # 🔁 Log each trial as child run
        # ----------------------------
        for i in range(len(results["params"])):
            with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
                params = results["params"][i]
                mean_score = results["mean_test_score"][i]
                std_score = results["std_test_score"][i]

                mlflow.log_params(params)
                mlflow.log_metric("cv_mean_rmse", -mean_score)
                mlflow.log_metric("cv_std_rmse", std_score)
                mlflow.set_tag("parent_run", parent_run.info.run_id)

        # ----------------------------
        # 🏆 Log Best Model and Metrics
        # ----------------------------
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        })

        # ----------------------------
        # 📊 Plots (log as artifacts)
        # ----------------------------
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3, s=1)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual EMI')
        plt.ylabel('Predicted EMI')
        plt.title('Actual vs Predicted EMI - Linear Regression')
        plt.tight_layout()
        act_pred_path = "linear_regression_actual_vs_pred.png"
        plt.savefig(act_pred_path, dpi=150)
        plt.close()
        mlflow.log_artifact(act_pred_path)
        os.remove(act_pred_path)

        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.3, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted EMI')
        plt.ylabel('Residuals')
        plt.title('Residual Plot - Linear Regression')
        plt.tight_layout()
        residuals_path = "linear_regression_residuals.png"
        plt.savefig(residuals_path, dpi=150)
        plt.close()
        mlflow.log_artifact(residuals_path)
        os.remove(residuals_path)

        # ----------------------------
        # 📈 Feature Importance
        # ----------------------------
        feature_names = (
            numerical_cols +
            list(best_model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_cols))
        )

        coefs = best_model.named_steps['regressor'].coef_
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        }).sort_values('Coefficient', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
        plt.title("Top 20 Coefficients - Linear Regression")
        plt.tight_layout()
        coef_path = "linear_regression_coefficients.png"
        plt.savefig(coef_path, dpi=150)
        plt.close()
        mlflow.log_artifact(coef_path)
        os.remove(coef_path)

        # ----------------------------
        # 💾 Save and log model
        # ----------------------------
        joblib.dump(best_model, "best_linear_regressor.pkl")
        mlflow.log_artifact("best_linear_regressor.pkl")
        signature = infer_signature(X_train, best_model.predict(X_train[:5]))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="best_linear_regressor",
            signature=signature,
            input_example=X_test.iloc[:2]
        )

        # ----------------------------
        # 🏷️ Tags and Metadata
        # ----------------------------
        mlflow.set_tag("mlflow.runName", "linear_regression_random_search")
        mlflow.set_tag("release.version", "1.0.0")
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("git.commit", git_commit)
        mlflow.set_tag("python.version", sys.version.split()[0])
        mlflow.log_artifact(__file__)

        print(f"\n✅ Best Parameters: {random_search.best_params_}")
        print(f"📈 RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}")
        print(f"🔖 Git Commit: {git_commit}")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")
        print("🎉 Linear Regression training and MLflow logging completed successfully!")


if __name__ == "__main__":
    main()
