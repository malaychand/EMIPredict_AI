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

    X = df.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y = df["emi_eligibility"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    label_mapping = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    print("Label Mapping:", label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"📊 Data split: Train={X_train.shape}, Test={X_test.shape}")

    # ----------------------------
    # Preprocessing pipeline
    # ----------------------------
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # ----------------------------
    # Model + RandomizedSearchCV
    # ----------------------------
    log_reg = LogisticRegression(max_iter=2000, solver="saga", multi_class="multinomial")

    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", log_reg)])

    param_dist = {
        "classifier__C": np.logspace(-3, 2, 10),
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__l1_ratio": np.linspace(0, 1, 5),
    }

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=5,
        scoring="f1_weighted",
        verbose=2,
        random_state=42,
        cv=3,
        n_jobs=-1,
        return_train_score=True
    )

    # ----------------------------
    # Parent MLflow run
    # ----------------------------
    with mlflow.start_run(run_name="logistic") as parent_run:
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dataset", os.path.basename(data_path))
        mlflow.set_tag("model", "LogisticRegression")
        mlflow.set_tag("search_type", "randomized")
        mlflow.set_tag("task", "classification")

        print("\n🔍 Running RandomizedSearchCV (this may take a bit)...")
        random_search.fit(X_train, y_train)

        # Log CV results as table (no local file)
        cv_results = pd.DataFrame(random_search.cv_results_)
        mlflow.log_table(data=cv_results, artifact_file="logistic_cv_results.json")

        # Log child runs
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"iteration_{idx+1}", nested=True):
                params = {
                    k.replace("param_", ""): cv_results.loc[idx, k]
                    for k in cv_results.columns if k.startswith("param_")
                }
                mlflow.log_params(params)

                for key in ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "rank_test_score"]:
                    if key in cv_results.columns:
                        val = cv_results.loc[idx, key]
                        if pd.notna(val):
                            mlflow.log_metric(key, float(val))

                split_keys = [c for c in cv_results.columns if c.startswith("split") and c.endswith("_test_score")]
                for sk in split_keys:
                    val = cv_results.loc[idx, sk]
                    if pd.notna(val):
                        mlflow.log_metric(sk, float(val))

                mlflow.set_tag("iteration", idx + 1)
                mlflow.set_tag("search_type", "randomized")

        # ----------------------------
        # Best model evaluation
        # ----------------------------
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_index = random_search.best_index_

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_score", float(random_search.best_score_))
        mlflow.log_metric("best_iteration_index", int(best_index))

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        y_pred_labels = label_enc.inverse_transform(y_pred)
        y_test_labels = label_enc.inverse_transform(y_test)

        acc = accuracy_score(y_test_labels, y_pred_labels)
        prec = precision_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
        rec = recall_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except Exception:
            classes = np.arange(len(label_enc.classes_))
            y_test_bin = label_binarize(y_test, classes=classes)
            roc_auc = roc_auc_score(y_test_bin, y_proba, average="macro")

        # ✅ Keep metric names EXACTLY as requested
        mlflow.log_metrics({
            "test_accuracy": float(acc),
            "test_precision_weighted": float(prec),
            "test_recall_weighted": float(rec),
            "test_f1_weighted": float(f1),
            "test_roc_auc_ovr": float(roc_auc)
        })

        # Classification report
        class_report = classification_report(y_test_labels, y_pred_labels, digits=4)
        print("\n=== Classification Report ===")
        print(class_report)

        report_text = f"Label Mapping:\n{label_mapping}\n\n{class_report}"
        mlflow.log_text(report_text, "classification_report.txt")

        # Confusion Matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_enc.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Logistic Regression")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # ROC Curves (One-vs-Rest)
        classes = np.arange(len(label_enc.classes_))
        y_test_bin = label_binarize(y_test, classes=classes)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(label_enc.classes_):
            try:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f"{class_name} (AUC = {auc_score:.3f})")
            except Exception:
                continue
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves (One-vs-Rest)")
        ax_roc.legend(loc="lower right", fontsize="small")
        mlflow.log_figure(fig_roc, "roc_curves.png")
        plt.close(fig_roc)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/logistic_classifier_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Final tags
        mlflow.set_tag("total_iterations", len(cv_results))
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")

        # Summary
        print("\n✅ Logistic Regression (RandomizedSearchCV) Completed")
        print(f"🏆 Best Params: {best_params}")
        print(f"📈 Best CV Score (f1_weighted): {random_search.best_score_:.4f}")
        print("📊 Test Metrics:")
        print(f"   - Accuracy:  {acc:.4f}")
        print(f"   - Precision: {prec:.4f}")
        print(f"   - Recall:    {rec:.4f}")
        print(f"   - F1 Score:  {f1:.4f}")
        print(f"   - ROC-AUC (OVR): {roc_auc:.4f}")
        print(f"\n🔗 MLflow URL: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()