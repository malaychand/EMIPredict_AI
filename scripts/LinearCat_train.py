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

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

from data_preprocessing import load_and_preprocess_data

warnings.filterwarnings("ignore")


# ----------------------------
# Helper: Get Git Commit Hash
# ----------------------------
def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    # ✅ Connect MLflow (DagsHub)
    dagshub.init(repo_owner="malaychand", repo_name="EMIPredict_AI", mlflow=True)
    print("✅ Connected to DagsHub MLflow Tracking Server")

    git_commit = get_git_commit()
    mlflow.set_experiment("EMI_prediction_classification")

    # ----------------------------
    # Locate dataset
    # ----------------------------
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv",
    ]
    data_path = next((path for path in possible_paths if os.path.exists(path)), None)
    if data_path is None:
        print("❌ Dataset not found. Please provide path:")
        data_path = input("Dataset path: ").strip()
        if not os.path.exists(data_path):
            sys.exit(f"❌ File not found at {data_path}")

    # ----------------------------
    # Load and preprocess
    # ----------------------------
    df = load_and_preprocess_data(data_path)

    print("\n📌 Original Class Distribution:")
    print(df["emi_eligibility"].value_counts())

    X = df.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y = df["emi_eligibility"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # ----------------------------
    # Encode labels
    # ----------------------------
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    label_mapping = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    print("\nLabel Mapping:", label_mapping)

    # ----------------------------
    # Preprocessing pipeline (WITHOUT applying fit_transform yet)
    # ----------------------------
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    # Fit only to transform training data later
    X_preprocessed = preprocessor.fit_transform(X)

    # ----------------------------
    # SMOTE Oversampling AFTER preprocessing
    # ----------------------------
    print("\n🔄 Applying SMOTE Oversampling to balance classes...")

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_preprocessed, y_encoded)

    print("\n📌 New Class Distribution After SMOTE:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"{label_enc.inverse_transform([cls])[0]} : {cnt}")

    # ----------------------------
    # Train-test split (balanced data)
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    print(f"\n📊 Data split: Train={X_train.shape}, Test={X_test.shape}")

    # ----------------------------
    # Model training: Logistic Regression + RandomizedSearchCV
    # ----------------------------
    log_reg = LogisticRegression(max_iter=2000, solver="saga", multi_class="multinomial")

    pipe = Pipeline([("classifier", log_reg)])

    param_dist = {
        "classifier__C": np.logspace(-3, 2, 10),
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__l1_ratio": np.linspace(0, 1, 5),
    }

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=4,
        scoring="f1_weighted",
        verbose=2,
        random_state=42,
        cv=3,
        n_jobs=-1,
        return_train_score=True
    )

    # ----------------------------
    # MLflow logging
    # ----------------------------
    with mlflow.start_run(run_name="logistic_smote") as parent_run:
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("balanced", "SMOTE")
        mlflow.set_tag("model", "LogisticRegression")

        # Log class distribution
        mlflow.log_text(str(counts.tolist()), "class_distribution_after_smote.txt")

        print("\n🔍 Running RandomizedSearchCV (this may take a bit)...")
        random_search.fit(X_train, y_train)

        # Save CV results
        cv_results = pd.DataFrame(random_search.cv_results_)
        mlflow.log_table(cv_results, "logistic_cv_results.json")

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        mlflow.log_params(best_params)

        # ----------------------------
        # Model Evaluation
        # ----------------------------
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        y_pred_labels = label_enc.inverse_transform(y_pred)
        y_test_labels = label_enc.inverse_transform(y_test)

        acc = accuracy_score(y_test_labels, y_pred_labels)
        prec = precision_score(y_test_labels, y_pred_labels, average="weighted")
        rec = recall_score(y_test_labels, y_pred_labels, average="weighted")
        f1 = f1_score(y_test_labels, y_pred_labels, average="weighted")

        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision_weighted": prec,
            "test_recall_weighted": rec,
            "test_f1_weighted": f1,
            "test_roc_auc_ovr": roc_auc
        })

        # Classification report
        report = classification_report(y_test_labels, y_pred_labels)
        mlflow.log_text(report, "classification_report.txt")

        # Save model (preprocessor + classifier)
        final_model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", best_model.named_steps["classifier"])
        ])

        os.makedirs("models", exist_ok=True)
        model_path = "models/logistic_classifier_smote.pkl"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(model_path)

        print("\n🎉 Model Training Complete with SMOTE Oversampling")
        print(f"🏆 Best Params: {best_params}")
        print(f"📈 F1 Score: {f1:.4f}")
        print(f"🔗 MLflow: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")


if __name__ == "__main__":
    main()
