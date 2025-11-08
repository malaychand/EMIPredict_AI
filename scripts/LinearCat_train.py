# =====================================================
# EMI Eligibility Prediction using Logistic Regression
# with RandomizedSearchCV for Hyperparameter Tuning
# =====================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression

# =====================================================
# Step 1: Prepare Data
# =====================================================
from data_preprocessing import load_and_preprocess_data
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

# Separate features and target
X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
y = df['emi_eligibility']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Encode target
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)
print("Label Mapping:", dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_))))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =====================================================
# Step 2: Preprocessing
# =====================================================
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# =====================================================
# Step 3: Logistic Regression Model + RandomizedSearchCV
# =====================================================
log_reg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='saga')

# Define parameter grid for random search
param_dist = {
    'classifier__C': np.logspace(-3, 2, 10),          # Regularization strength
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
    'classifier__l1_ratio': np.linspace(0, 1, 5)       # Only used if elasticnet
}

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', log_reg)])

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=2,
    scoring='f1_weighted',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# =====================================================
# Step 4: Train Model
# =====================================================
print("🔍 Running RandomizedSearchCV for Logistic Regression...")
random_search.fit(X_train, y_train)

print("\n✅ Best Parameters Found:")
print(random_search.best_params_)

# =====================================================
# Step 5: Evaluate Model
# =====================================================
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_labels = label_enc.inverse_transform(y_pred)
y_test_labels = label_enc.inverse_transform(y_test)

print("\n=== Classification Report ===")
print(classification_report(y_test_labels, y_pred_labels))

acc = accuracy_score(y_test_labels, y_pred_labels)
prec = precision_score(y_test_labels, y_pred_labels, average='weighted')
rec = recall_score(y_test_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

# =====================================================
# Step 6: Confusion Matrix
# =====================================================
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_enc.classes_)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================================
# Step 7: ROC-AUC (Multi-class)
# =====================================================
y_proba = best_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
print(f"Multiclass ROC-AUC Score: {roc_auc:.4f}")
