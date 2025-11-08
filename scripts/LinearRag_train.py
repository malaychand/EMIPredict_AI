# linear_regression_tuning.py

import os
import sys
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from data_preprocessing import load_and_preprocess_data
warnings.filterwarnings('ignore')


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
    # Model setup (try Ridge, Lasso, ElasticNet)
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

    print("\n🔍 Starting RandomizedSearchCV (3-fold CV, 25 iterations)...")
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=3,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("\n✅ Best Parameters:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")

    best_model = random_search.best_estimator_

    # ==============================
    # Evaluation
    # ==============================
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("\n📈 Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")

    # ==============================
    # Plots
    # ==============================
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual EMI')
    plt.ylabel('Predicted EMI')
    plt.title('Actual vs Predicted EMI - Linear Regression')
    plt.tight_layout()
    plt.savefig("linear_regression_actual_vs_pred.png", dpi=150)

    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted EMI')
    plt.ylabel('Residuals')
    plt.title('Residual Plot - Linear Regression')
    plt.tight_layout()
    plt.savefig("linear_regression_residuals.png", dpi=150)

    # ==============================
    # Feature Importance (Coefficients)
    # ==============================
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
    plt.savefig("linear_regression_coefficients.png", dpi=150)

    # ==============================
    # Save model
    # ==============================
    joblib.dump(best_model, "best_linear_regressor.pkl")
    print("\n💾 Model saved as 'best_linear_regressor.pkl'")
    print("🎉 Linear Regression training completed successfully!")


if __name__ == "__main__":
    main()
