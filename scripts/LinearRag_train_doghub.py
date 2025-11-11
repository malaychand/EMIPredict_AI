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
from sklearn.ensemble import RandomForestRegressor

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
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv"
    ]
    data_path = next((path for path in possible_paths if os.path.exists(path)), None)
    if data_path is None:
        sys.exit("❌ Dataset not found. Please verify path.")

    # ✅ Load and preprocess
    df = load_and_preprocess_data(data_path)
    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['max_monthly_emi']

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n📊 Features: {len(numerical_cols)} numeric, {len(categorical_cols)} categorical")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🧩 Split: Train={X_train.shape}, Test={X_test.shape}")

    # ✅ Pipeline setup
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    rf = RandomForestRegressor(random_state=42)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])

    # ✅ Hyperparameter tuning space
    param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [5, 10, 15, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2'],
        'regressor__bootstrap': [True, False]
    }

    print("\n🔍 Starting RandomizedSearchCV (3-fold CV, 4 iterations)...")
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=4,
        scoring='r2',
        verbose=2,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    git_commit = get_git_commit()
    mlflow.set_experiment("emi_prediction_regression")

    # ✅ Parent run for all iterations
    with mlflow.start_run(run_name="random_forest_regression_randomsearch_parent") as parent_run:
        random_search.fit(X_train, y_train)
        cv_results = pd.DataFrame(random_search.cv_results_)

        # ✅ Log all iterations as child runs
        print("\n📊 Logging all RandomizedSearchCV iterations to MLflow...")
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"iteration_{idx+1}", nested=True) as child_run:
                params = {k.replace('param_', ''): cv_results.loc[idx, k]
                          for k in cv_results.columns if k.startswith('param_')}
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "mean_train_score": cv_results.loc[idx, 'mean_train_score'],
                    "mean_test_score": cv_results.loc[idx, 'mean_test_score'],
                    "rank_test_score": cv_results.loc[idx, 'rank_test_score']
                })
                for fold in range(3):
                    mlflow.log_metric(f"split{fold}_test_score", cv_results.loc[idx, f'split{fold}_test_score'])
                mlflow.set_tag("iteration", idx + 1)

        # ✅ Best model evaluation
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_index = random_search.best_index_

        y_pred = best_model.predict(X_test)

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

        # ==============================
        # 📊 Visualization: Actual vs Predicted
        # ==============================
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual EMI")
        plt.ylabel("Predicted EMI")
        plt.title("Actual vs Predicted EMI - Random Forest")
        plt.tight_layout()
        act_pred_path = "rf_regression_actual_vs_pred.png"
        plt.savefig(act_pred_path, dpi=150)
        plt.close()
        mlflow.log_artifact(act_pred_path)
        os.remove(act_pred_path)

        # ==============================
        # 📊 Residual Plot
        # ==============================
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.4)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted EMI")
        plt.ylabel("Residuals")
        plt.title("Residual Plot - Random Forest")
        plt.tight_layout()
        residuals_path = "rf_regression_residuals.png"
        plt.savefig(residuals_path, dpi=150)
        plt.close()
        mlflow.log_artifact(residuals_path)
        os.remove(residuals_path)

        # ==============================
        # 📊 Feature Importance
        # ==============================
        feature_names = (
            numerical_cols +
            list(best_model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_cols))
        )

        importances = best_model.named_steps['regressor'].feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp)
        plt.title("Top 20 Feature Importances - Random Forest Regressor")
        plt.tight_layout()
        feat_imp_path = "rf_regression_feature_importance.png"
        plt.savefig(feat_imp_path, dpi=150)
        plt.close()
        mlflow.log_artifact(feat_imp_path)
        os.remove(feat_imp_path)

        # ==============================
        # 📊 Save CV Results
        # ==============================
        cv_results_path = "rf_randomsearch_cv_results.csv"
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)
        os.remove(cv_results_path)

        # ==============================
        # 💾 Save Model & Signature
        # ==============================
        joblib.dump(best_model, "best_rf_regressor.pkl")
        mlflow.log_artifact("best_rf_regressor.pkl")

        signature = infer_signature(X_train, best_model.predict(X_train[:5]))
        with open("rf_regression_signature.txt", "w") as f:
            f.write(str(signature))
        mlflow.log_artifact("rf_regression_signature.txt")
        os.remove("rf_regression_signature.txt")

        # ==============================
        # 🏷 Metadata
        # ==============================
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("model", "RandomForestRegressor")
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("search_type", "randomized")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("total_iterations", len(cv_results))
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")

        print(f"\n✅ Random Forest Regression Completed")
        print(f"🏆 Best Params: {best_params}")
        print(f"📈 Best CV Score (R²): {random_search.best_score_:.4f}")
        print(f"📊 Test Metrics:")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MAE:  {mae:.2f}")
        print(f"   - R²:   {r2:.4f}")
        print(f"   - MAPE: {mape:.4f}")
        print(f"\n🔗 Run URL: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
