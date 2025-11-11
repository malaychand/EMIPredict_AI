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

    # ✅ Model pipeline with Ridge (supports alpha parameter)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    # ✅ Define hyperparameter space for RandomizedSearchCV
    param_distributions = {
        'regressor__alpha': uniform(0.01, 100),
        'regressor__fit_intercept': [True, False],
        'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
        'regressor__max_iter': [1000, 2000, 5000]
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
        return_train_score=True  # Important: Return training scores for comparison
    )

    git_commit = get_git_commit()
    mlflow.set_experiment("emi_prediction_regression")

    # ✅ Parent run for the entire search
    with mlflow.start_run(run_name="linear_regression_randomsearch_parent") as parent_run:
        
        # ✅ Fit Randomized Search
        print("\n🔍 Starting RandomizedSearchCV with Ridge Regression...")
        random_search.fit(X_train, y_train)

        # ✅ Extract results from CV
        cv_results = pd.DataFrame(random_search.cv_results_)
        
        # ✅ Log each iteration as a child run
        print("\n📊 Logging all RandomizedSearchCV iterations...")
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"iteration_{idx+1}", nested=True) as child_run:
                # Extract parameters for this iteration
                params = {}
                for key in cv_results.columns:
                    if key.startswith('param_'):
                        param_name = key.replace('param_', '')
                        params[param_name] = cv_results.loc[idx, key]
                
                # Log parameters
                mlflow.log_params(params)
                
                # Log CV metrics
                mlflow.log_metrics({
                    "mean_test_score": cv_results.loc[idx, 'mean_test_score'],
                    "std_test_score": cv_results.loc[idx, 'std_test_score'],
                    "mean_train_score": cv_results.loc[idx, 'mean_train_score'],
                    "std_train_score": cv_results.loc[idx, 'std_train_score'],
                    "mean_fit_time": cv_results.loc[idx, 'mean_fit_time'],
                    "rank_test_score": cv_results.loc[idx, 'rank_test_score']
                })
                
                # Log individual fold scores
                for fold in range(3):  # cv=3
                    mlflow.log_metric(f"split{fold}_test_score", 
                                    cv_results.loc[idx, f'split{fold}_test_score'])
                    mlflow.log_metric(f"split{fold}_train_score", 
                                    cv_results.loc[idx, f'split{fold}_train_score'])
                
                mlflow.set_tag("iteration", idx + 1)
                mlflow.set_tag("search_type", "randomized")

        # ✅ Extract best model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_index = random_search.best_index_

        # ✅ Predictions on test set
        y_pred = best_model.predict(X_test)

        # ✅ Evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # ✅ Log best model parameters and metrics to parent run
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "best_cv_score": random_search.best_score_,
            "best_iteration": best_index + 1
        })

        # ✅ Create comparison visualization of all iterations
        plt.figure(figsize=(12, 6))
        
        # Plot 1: R² scores across iterations
        plt.subplot(1, 2, 1)
        iterations = range(1, len(cv_results) + 1)
        plt.plot(iterations, cv_results['mean_test_score'], 'o-', label='Mean Test Score', linewidth=2)
        plt.fill_between(iterations, 
                        cv_results['mean_test_score'] - cv_results['std_test_score'],
                        cv_results['mean_test_score'] + cv_results['std_test_score'],
                        alpha=0.2)
        plt.axhline(y=random_search.best_score_, color='r', linestyle='--', label='Best Score')
        plt.xlabel('Iteration')
        plt.ylabel('R² Score')
        plt.title('Cross-Validation Scores Across Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Train vs Test scores
        plt.subplot(1, 2, 2)
        plt.plot(iterations, cv_results['mean_train_score'], 'o-', label='Train Score', linewidth=2)
        plt.plot(iterations, cv_results['mean_test_score'], 's-', label='Test Score', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('R² Score')
        plt.title('Train vs Test Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        iterations_path = "randomsearch_iterations_comparison.png"
        plt.savefig(iterations_path, dpi=150)
        plt.close()
        mlflow.log_artifact(iterations_path)
        os.remove(iterations_path)

        # ✅ Save CV results as CSV
        cv_results_path = "randomsearch_cv_results.csv"
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)
        os.remove(cv_results_path)

        # ✅ Visualization: Actual vs Predicted (Best Model)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual EMI")
        plt.ylabel("Predicted EMI")
        plt.title("Actual vs Predicted EMI (Best Ridge Model)")
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
        plt.title("Residual Plot (Best Ridge Model)")
        plt.tight_layout()
        residuals_path = "linear_regression_residuals.png"
        plt.savefig(residuals_path, dpi=150)
        plt.close()
        mlflow.log_artifact(residuals_path)
        os.remove(residuals_path)

        # ✅ Parameter importance visualization
        plt.figure(figsize=(10, 6))
        
        # Extract parameter columns
        param_cols = [col for col in cv_results.columns if col.startswith('param_')]
        
        # Create a summary of how parameters affect performance
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            unique_vals = cv_results[param_col].unique()
            
            # Only plot if there are multiple unique values
            if len(unique_vals) > 1 and len(unique_vals) <= 10:
                scores_by_param = cv_results.groupby(param_col)['mean_test_score'].mean().sort_values()
                plt.figure(figsize=(8, 5))
                scores_by_param.plot(kind='barh')
                plt.xlabel('Mean R² Score')
                plt.title(f'Performance by {param_name}')
                plt.tight_layout()
                param_plot_path = f"param_importance_{param_name}.png"
                plt.savefig(param_plot_path, dpi=150)
                plt.close()
                mlflow.log_artifact(param_plot_path)
                os.remove(param_plot_path)

        # ✅ Save and Log Best Model
        joblib.dump(best_model, "linear_regression_model.pkl")
        mlflow.log_artifact("linear_regression_model.pkl")

        # ✅ Log model signature
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
        mlflow.set_tag("search_type", "randomized")
        mlflow.set_tag("total_iterations", len(cv_results))
        mlflow.log_artifact(__file__)

        # ✅ Print summary
        print(f"\n✅ Ridge Regression (RandomizedSearchCV) Completed")
        print(f"🔍 Total Iterations: {len(cv_results)}")
        print(f"🏆 Best Iteration: {best_index + 1}")
        print(f"🏆 Best Params: {best_params}")
        print(f"📈 Best CV Score (R²): {random_search.best_score_:.4f}")
        print(f"📊 Test Set Metrics:")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MAE: {mae:.2f}")
        print(f"   - R²: {r2:.4f}")
        print(f"   - MAPE: {mape:.2f}")
        print(f"\n🔗 Run URL: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")
        
        # Print iteration summary
        print(f"\n📋 Iteration Summary:")
        print(cv_results[['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_fit_time']].sort_values('rank_test_score'))


if __name__ == "__main__":
    main()