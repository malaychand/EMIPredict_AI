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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import uniform

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

    # ✅ Model pipeline with Linear Regression
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # ✅ Define hyperparameter space (only a few tunable options for LinearRegression)
    param_distributions = {
        'regressor__fit_intercept': [True, False],
        'regressor__positive': [True, False],
    }

    # ✅ Randomized Search CV
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=4,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=2,
        return_train_score=True
    )

    git_commit = get_git_commit()
    mlflow.set_experiment("emi_prediction_regression")

    # ✅ Parent run for all iterations
    with mlflow.start_run(run_name="linear_regression_randomsearch_parent") as parent_run:
        print("\n🔍 Starting RandomizedSearchCV with Linear Regression...")
        random_search.fit(X_train, y_train)
        cv_results = pd.DataFrame(random_search.cv_results_)

        # ✅ Log each iteration as child run
        print("\n📊 Logging all RandomizedSearchCV iterations...")
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"iteration_{idx+1}", nested=True) as child_run:
                params = {k.replace('param_', ''): cv_results.loc[idx, k] 
                          for k in cv_results.columns if k.startswith('param_')}
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "mean_test_score": cv_results.loc[idx, 'mean_test_score'],
                    "mean_train_score": cv_results.loc[idx, 'mean_train_score'],
                    "rank_test_score": cv_results.loc[idx, 'rank_test_score']
                })
                for fold in range(3):
                    mlflow.log_metric(f"split{fold}_test_score", cv_results.loc[idx, f'split{fold}_test_score'])
                mlflow.set_tag("iteration", idx + 1)

        # ✅ Best Model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_index = random_search.best_index_

        # ✅ Predictions
        y_pred = best_model.predict(X_test)

        # ✅ Evaluation Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "best_cv_score": random_search.best_score_,
            "best_iteration": best_index + 1
        })

        # ✅ Visualizations
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual EMI")
        plt.ylabel("Predicted EMI")
        plt.title("Actual vs Predicted EMI (Best Linear Regression Model)")
        plt.tight_layout()
        act_pred_path = "linear_regression_actual_vs_pred.png"
        plt.savefig(act_pred_path, dpi=150)
        plt.close()
        mlflow.log_artifact(act_pred_path)
        os.remove(act_pred_path)

        # Residual Plot
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

        # ✅ Save CV results
        cv_results_path = "linear_regression_cv_results.csv"
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)
        os.remove(cv_results_path)

        # ✅ Save model
        joblib.dump(best_model, "linear_regression_model.pkl")
        mlflow.log_artifact("linear_regression_model.pkl")

        # ✅ Log model signature
        signature = infer_signature(X_train, best_model.predict(X_train[:5]))
        with open("linear_regression_signature.txt", "w") as f:
            f.write(str(signature))
        mlflow.log_artifact("linear_regression_signature.txt")
        os.remove("linear_regression_signature.txt")

        # ✅ Metadata
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("model", "LinearRegression")
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")

        # ✅ Summary
        print(f"\n✅ Linear Regression (RandomizedSearchCV) Completed")
        print(f"🏆 Best Params: {best_params}")
        print(f"📈 Best CV Score (R²): {random_search.best_score_:.4f}")
        print(f"📊 Test Metrics:")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MAE: {mae:.2f}")
        print(f"   - R²: {r2:.4f}")
        print(f"   - MAPE: {mape:.2f}")
        print(f"\n🔗 Run URL: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
