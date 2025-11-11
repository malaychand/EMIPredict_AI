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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
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
# Load Dataset
# ----------------------------
def load_data():
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv"
    ]
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if data_path is None:
        sys.exit("❌ Dataset not found.")
    print(f"✅ Using dataset: {data_path}")
    return pd.read_csv(data_path)


# ----------------------------
# Main Training Function
# ----------------------------
def main():
    dagshub.init(repo_owner="malaychand", repo_name="emi-eligibility-mlflow", mlflow=True)
    print("✅ Connected to DagsHub MLflow Tracking Server")

    df = load_data()
    print("Dataset shape:", df.shape)

    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['emi_eligibility']

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    print("Label Mapping:", dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    numeric_tf = Pipeline([('scaler', StandardScaler())])
    categorical_tf = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num', numeric_tf, numerical_cols),
        ('cat', categorical_tf, categorical_cols)
    ])

    rf = RandomForestClassifier(random_state=42)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=2,
        scoring='f1_weighted',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    git_commit = get_git_commit()
    mlflow.set_experiment("RandomForest_Classification")

    with mlflow.start_run(run_name="RandomForest_Parent_Run") as parent_run:
        print("🌲 Running RandomizedSearchCV...")
        random_search.fit(X_train, y_train)
        results = pd.DataFrame(random_search.cv_results_)

        print("\n📊 Logging all RandomizedSearchCV iterations to MLflow...")
        for i in range(len(results)):
            with mlflow.start_run(run_name=f"Iteration_{i+1}", nested=True):
                params = {k.replace('param_', ''): results.loc[i, k]
                          for k in results.columns if k.startswith('param_')}
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "mean_train_score": results.loc[i, 'mean_train_score'],
                    "mean_test_score": results.loc[i, 'mean_test_score']
                })

        print("\n✅ Best Parameters Found:")
        print(random_search.best_params_)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=label_enc.classes_))
        print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics({
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "roc_auc": roc_auc
        })

        # ==============================
        # 📊 Confusion Matrix
        # ==============================
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
        plt.title("Confusion Matrix - Random Forest Classifier")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = "confusion_matrix_rf.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # ==============================
        # 💾 Save Model & Signature
        # ==============================
        joblib.dump(best_model, "best_rf_classifier.pkl")
        mlflow.log_artifact("best_rf_classifier.pkl")

        signature = infer_signature(X_train, best_model.predict(X_train[:5]))
        with open("rf_classifier_signature.txt", "w") as f:
            f.write(str(signature))
        mlflow.log_artifact("rf_classifier_signature.txt")
        os.remove("rf_classifier_signature.txt")

        # ==============================
        # 🏷 Metadata
        # ==============================
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("model", "RandomForestClassifier")
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("search_type", "randomized")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/emi-eligibility-mlflow.mlflow")

        print(f"\n✅ Random Forest Classification Completed Successfully")
        print(f"🏆 Best Params: {random_search.best_params_}")
        print(f"📈 Test Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        print(f"🔗 View on DagsHub: https://dagshub.com/malaychand/emi-eligibility-mlflow.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
