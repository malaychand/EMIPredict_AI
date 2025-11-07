# xgboost_regression_tuning.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor


def load_and_preprocess_data(data_path):
    """
    Load and engineer features from EMI dataset for regression
    """
    print(f"📥 Attempting to load dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with shape: {df.shape}")

    # Feature Engineering (as per your spec)
    df["debt_to_income"] = df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)
    
    expense_cols = [
        "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
        "other_monthly_expenses", "monthly_rent"
    ]
    df["total_monthly_expenses"] = df[expense_cols].sum(axis=1)
    df["expense_to_income"] = df["total_monthly_expenses"] / df["monthly_salary"].replace(0, np.nan)
    df["monthly_disposable"] = (
        df["monthly_salary"] - df["total_monthly_expenses"] - df["current_emi_amount"]
    )
    df["instalment_if_approved"] = df["requested_amount"] / df["requested_tenure"].replace(0, np.nan)
    df["affordability_ratio"] = df["monthly_disposable"] / df["instalment_if_approved"].replace(0, np.nan)
    df["employment_stability"] = df["years_of_employment"] / df["age"].replace(0, np.nan)
    df["loan_to_income_ratio"] = df["requested_amount"] / df["monthly_salary"].replace(0, np.nan)
    df["dependents_ratio"] = df["dependents"] / df["family_size"].replace(0, np.nan)
    
    # Clean infinities and NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['max_monthly_emi'], inplace=True)  # Critical: Keep only valid targets
    df.dropna(inplace=True)  # Remove rows with any NaN after engineering
    
    print(f"🧹 After cleaning: {df.shape}")
    return df


def main():
    # ==============================
    # CONFIGURATION - MODIFY THESE PATHS AS NEEDED
    # ==============================
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("\n❌ Dataset not found in common locations!")
        print("Please provide the full path to your 'cleaned_EMI_dataset.csv':")
        data_path = input("Dataset path: ").strip()
        if not os.path.exists(data_path):
            print(f"❌ File still not found: {data_path}")
            sys.exit(1)
    
    # Load and preprocess
    df = load_and_preprocess_data(data_path)
    
    # ==============================
    # FEATURE/TARGET SETUP
    # ==============================
    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['max_monthly_emi']
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n📊 Features detected:")
    print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical ({len(numerical_cols)}): {numerical_cols}")
    print(f"\n📈 Target stats:\n{y.describe()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nSplitOptions: Train={X_train.shape}, Test={X_test.shape}")
    
    # ==============================
    # PREPROCESSING PIPELINE
    # ==============================
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # ==============================
    # XGBOOST REGRESSOR + RANDOMIZED SEARCH
    # ==============================
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb)
    ])
    
    # Hyperparameter grid for regression
    param_dist = {
        'regressor__n_estimators': [100, 200, 300, 400],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.15],
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__colsample_bytree': [0.6, 0.8, 1.0],
        'regressor__gamma': [0, 0.1, 0.3],
        'regressor__reg_alpha': [0, 0.5, 1.0],      # L1 regularization
        'regressor__reg_lambda': [1, 1.5, 2.0]      # L2 regularization
    }
    
    # RandomizedSearchCV
    print("\n🔍 Starting RandomizedSearchCV (3-fold CV, 25 iterations)...")
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=2,
        scoring='neg_root_mean_squared_error',  # Optimize for lowest RMSE
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\n" + "="*60)
    print("✅ BEST PARAMETERS:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # ==============================
    # EVALUATION
    # ==============================
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print("\n" + "="*60)
    print("📈 MODEL PERFORMANCE:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")
    
    # Prediction vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Max Monthly EMI')
    plt.ylabel('Predicted Max Monthly EMI')
    plt.title('Actual vs Predicted EMI Amounts')
    plt.tight_layout()
    plt.savefig("regression_actual_vs_pred.png", dpi=150)
    print("\n💾 Actual vs Predicted plot saved as 'regression_actual_vs_pred.png'")
    
    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted EMI')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig("regression_residuals.png", dpi=150)
    print("💾 Residuals plot saved as 'regression_residuals.png'")
    
    # Feature Importance
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
    plt.title("Top 20 Feature Importances - XGBoost Regressor")
    plt.tight_layout()
    plt.savefig("regression_feature_importance.png", dpi=150)
    print("💾 Feature importance plot saved as 'regression_feature_importance.png'")
    
    # Save model
    import joblib
    joblib.dump(best_model, "best_xgb_regressor.pkl")
    print("\n💾 Best model saved as 'best_xgb_regressor.pkl'")
    print("\n🎉 Regression training completed successfully!")


if __name__ == "__main__":
    main()