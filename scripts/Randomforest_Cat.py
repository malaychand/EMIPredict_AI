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


def log_plot_to_mlflow(fig, artifact_name):
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)


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
    # ✅ Connect MLflow (DagsHub)
    dagshub.init(repo_owner="malaychand", repo_name="EMIPredict_AI", mlflow=True)
    print("✅ Connected to DagsHub MLflow Tracking Server")

    git_commit = get_git_commit()
    mlflow.set_experiment("EMI_prediction_classification")

    df = load_data()
    print("Dataset shape:", df.shape)

    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['emi_eligibility']

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    label_mapping = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    print("Label Mapping:", label_mapping)

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
        n_iter=6,
        scoring='f1_weighted',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    with mlflow.start_run(run_name="RandomForest") as parent_run:
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
                    "mean_train_score": float(results.loc[i, 'mean_train_score']),
                    "mean_test_score": float(results.loc[i, 'mean_test_score'])
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

        class_report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
        print("\n=== Classification Report ===")
        print(class_report)

        # Key Performance Metrics (no "test_" prefix)
        mlflow.log_metrics({
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        })

        # Classification report as text artifact
        mlflow.log_text(
            "Label Mapping:\n" + str(label_mapping) + "\n\n" + class_report,
            "classification_report.txt"
        )

        # Confusion Matrix (log as figure)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, ax=ax)
        ax.set_title("Confusion Matrix - Random Forest Classifier")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        log_plot_to_mlflow(fig, "confusion_matrix_rf.png")

        # Save model to models/ and log
        os.makedirs("models", exist_ok=True)
        model_path = "models/rf_classifier.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Metadata tags
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("model", "RandomForestClassifier")
        mlflow.set_tag("dataset", "cleaned_EMI_dataset")
        mlflow.set_tag("search_type", "randomized")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")

        # Final summary
        print(f"\n✅ Random Forest Classification Completed Successfully")
        print(f"🏆 Best Params: {random_search.best_params_}")
        print(f"📊 Key Performance Metrics:")
        print(f"   - Accuracy:  {acc:.4f}")
        print(f"   - Precision: {prec:.4f}")
        print(f"   - Recall:    {rec:.4f}")
        print(f"   - F1 Score:  {f1:.4f}")
        print(f"   - ROC-AUC:   {roc_auc:.4f}")
        print(f"\n🔗 View on DagsHub: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()