# linear_regression_tuning_mlflow.py

import os
import sys
import warnings
import joblib
import subprocess
import dagshub
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from data_preprocessing import load_and_preprocess_data

warnings.filterwarnings('ignore')


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def main():
    # ✅ Initialize DagsHub + MLflow
    dagshub.init(repo_owner='malaychand', repo_name='EMIPredict_AI', mlflow=True)
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
    # Preprocessing Pipeline
    # ==============================
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    lr = LinearRegression()
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lr)
    ])

    # ==============================
    # Minimal hyperparameter space (for consistency)
    # ==============================
    param_dist = {
        'regressor__fit_intercept': [True, False],
        'regressor__positive': [True, False]
    }

    print("\n🔍 Starting RandomizedSearchCV (3-fold CV, 4 iterations)...")
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=2,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        random_state=42,
        n_jobs=-1,
        cv=3,
        return_train_score=True
    )

    # ==============================
    # MLflow: Parent + Child Runs
    # ==============================
    with mlflow.start_run(run_name="linear_regression_parent") as parent_run:
        random_search.fit(X_train, y_train)
        cv_results = pd.DataFrame(random_search.cv_results_)

        # ✅ Log each iteration as a nested child run
        print("\n📊 Logging all CV iterations as child runs...")
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"lr_iteration_{idx+1}", nested=True) as child_run:
                params = {
                    k.replace('param_', ''): cv_results.loc[idx, k]
                    for k in cv_results.columns if k.startswith('param_')
                }
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_rmse", -cv_results.loc[idx, 'mean_test_score'])
                mlflow.set_tag("iteration", idx + 1)

        # ✅ Final evaluation
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

        # ==============================
        # Visualizations → Direct to MLflow
        # ==============================
        # 1. Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual EMI')
        plt.ylabel('Predicted EMI')
        plt.title('Actual vs Predicted EMI - Linear Regression')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "figures/lr_actual_vs_pred.png")
        plt.close()

        # 2. Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.4)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted EMI')
        plt.ylabel('Residuals')
        plt.title('Residual Plot - Linear Regression')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "figures/lr_residuals.png")
        plt.close()

        # 3. Feature Coefficients (as importance proxy)
        feature_names = (
            numerical_cols +
            list(
                best_model.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(categorical_cols)
            )
        )
        coef = best_model.named_steps['regressor'].coef_
        feat_coef = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef
        })
        feat_coef['AbsCoefficient'] = feat_coef['Coefficient'].abs()
        top_feat = feat_coef.nlargest(20, 'AbsCoefficient')

        plt.figure(figsize=(10, 8))
        sns.barplot(x='AbsCoefficient', y='Feature', data=top_feat)
        plt.title("Top 20 Features by |Coefficient| - Linear Regression")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "figures/lr_feature_importance.png")
        plt.close()

        # ==============================
        # 💾 Save model in `models/` dir
        # ==============================
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_linear_regressor.pkl"

        # Remove existing file if it exists
        if os.path.exists(model_path):
            os.remove(model_path)

        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Log signature
        signature = infer_signature(X_train, best_model.predict(X_train[:5]))
        mlflow.log_text(str(signature), "model_signature.txt")

        # ==============================
        # Metadata
        # ==============================
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("model", "LinearRegression")
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")

        print("\n✅ Linear Regression Tuning Completed")
        print(f"🏆 Best Params: {random_search.best_params_}")
        print(f"📊 Test Metrics:")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MAE:  {mae:.2f}")
        print(f"   - R²:   {r2:.4f}")
        print(f"   - MAPE: {mape:.4f} ({mape*100:.2f}%)")
        print(f"\n🔗 MLflow Run: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()