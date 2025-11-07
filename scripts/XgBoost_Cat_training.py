# xgboost_classification_tuning.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def load_and_preprocess_data(data_path):
    """
    Load and engineer features from EMI dataset
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
    df.dropna(inplace=True)  # Critical: Remove rows with NaN after feature engineering
    
    print(f"🧹 After cleaning: {df.shape}")
    return df


def main():
    # ==============================
    # CONFIGURATION - MODIFY THESE PATHS AS NEEDED
    # ==============================
    # Option 1: Try common locations
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
    
    # Option 2: If not found, require user to specify
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
    y = df['emi_eligibility']
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n📊 Features detected:")
    print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical ({len(numerical_cols)}): {numerical_cols}")
    
    # Encode target
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    print(f"\n🏷️  Target classes: {dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
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
    # XGBOOST + RANDOMIZED SEARCH
    # ==============================
    xgb = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False
    )
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])
    
    # Hyperparameter grid
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [4, 6, 8],
        'classifier__learning_rate': [0.05, 0.1, 0.15],
        'classifier__subsample': [0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__min_child_weight': [1, 3]
    }
    
    # RandomizedSearchCV
    print("\n🔍 Starting RandomizedSearchCV (3-fold CV, 20 iterations)...")
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=2,
        scoring='f1_weighted',
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
    
    # Decode labels for reporting
    y_test_labels = label_enc.inverse_transform(y_test)
    y_pred_labels = label_enc.inverse_transform(y_pred)
    
    # Metrics
    print("\n" + "="*60)
    print("📈 MODEL PERFORMANCE:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    acc = accuracy_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (Weighted): {f1:.4f}")
    print(f"ROC-AUC (OvR): {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_enc.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
    plt.title("Confusion Matrix - XGBoost Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix_xgb.png", dpi=150)
    print("\n💾 Confusion matrix saved as 'confusion_matrix_xgb.png'")
    
    # Feature Importance
    feature_names = (
        numerical_cols +
        list(best_model.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_cols))
    )
    
    importances = best_model.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance_xgb.png", dpi=150)
    print("💾 Feature importance plot saved as 'feature_importance_xgb.png'")
    
    # Save model
    import joblib
    joblib.dump(best_model, "best_xgb_classifier.pkl")
    print("\n💾 Best model saved as 'best_xgb_classifier.pkl'")
    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    main()