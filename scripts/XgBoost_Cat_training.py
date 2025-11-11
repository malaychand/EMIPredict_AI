# xgboost_classification_tuning_mlflow_fixed.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
from xgboost import XGBClassifier
from data_preprocessing import load_and_preprocess_data

# MLflow + DagsHub
import mlflow
import dagshub
from mlflow.models.signature import infer_signature

def main():
    # ------------------------
    # DagsHub + MLflow setup
    # ------------------------
    # Edit repo_owner/repo_name if needed
    dagshub.init(repo_owner="malaychand", repo_name="emi-eligibility-mlflow", mlflow=True)
    mlflow.set_experiment("XGBoost_EMI_Classification")

    # ------------------------
    # Locate dataset
    # ------------------------
    possible_paths = [
        "data/cleaned_EMI_dataset.csv",
        "../data/cleaned_EMI_dataset.csv",
        "cleaned_EMI_dataset.csv",
        "./data/cleaned_EMI_dataset.csv"
    ]
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if data_path is None:
        print("❌ Dataset not found. Please provide path:")
        data_path = input("Dataset path: ").strip()
        if not os.path.exists(data_path):
            sys.exit(f"❌ File not found at {data_path}")

    # ------------------------
    # Load + preprocess
    # ------------------------
    df = load_and_preprocess_data(data_path)

    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['emi_eligibility']

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ------------------------
    # Preprocessing pipeline
    # ------------------------
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # ------------------------
    # Model + RandomizedSearchCV
    # ------------------------
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

    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [4, 6, 8],
        'classifier__learning_rate': [0.05, 0.1, 0.15],
        'classifier__subsample': [0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__min_child_weight': [1, 3]
    }

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=3,
        scoring='f1_weighted',
        verbose=2,
        cv=3,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    # ------------------------
    # Parent MLflow run
    # ------------------------
    with mlflow.start_run(run_name="XGBoost_Parent_Run") as parent_run:
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", "unknown")  # replace with git commit if available
        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("search_type", "RandomizedSearchCV")
        mlflow.log_param("n_iter", 3)
        mlflow.log_params({f"param_dist__{k}": str(v) for k, v in param_dist.items()})

        print("\n🔍 Running RandomizedSearchCV...")
        random_search.fit(X_train, y_train)

        print("\n✅ Best Parameters Found:")
        print(random_search.best_params_)
        mlflow.log_params({f"best_{k}": v for k, v in random_search.best_params_.items()})
        mlflow.log_metric("best_cv_score", float(random_search.best_score_))

        # Save CV results as artifact
        cv_results = pd.DataFrame(random_search.cv_results_)
        cv_path = "xgb_cv_results.csv"
        cv_results.to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path)
        try:
            os.remove(cv_path)
        except Exception:
            pass

        # Log each candidate (child runs)
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"candidate_{idx+1}", nested=True):
                params = {k.replace("param_", ""): cv_results.loc[idx, k]
                          for k in cv_results.columns if k.startswith("param_")}
                if params:
                    mlflow.log_params(params)
                # log common cv metrics if present
                for key in ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "rank_test_score"]:
                    if key in cv_results.columns:
                        try:
                            mlflow.log_metric(key, float(cv_results.loc[idx, key]))
                        except Exception:
                            pass

        # ------------------------
        # Best model evaluation on holdout set
        # ------------------------
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # decode labels for readable reports
        y_test_labels = label_enc.inverse_transform(y_test)
        y_pred_labels = label_enc.inverse_transform(y_pred)

        acc = accuracy_score(y_test_labels, y_pred_labels)
        prec = precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        rec = recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)

        # ROC-AUC (multi-class OVR)
        try:
            y_proba = best_model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except Exception:
            roc_auc = np.nan

        mlflow.log_metrics({
            "test_accuracy": float(acc),
            "test_precision_weighted": float(prec),
            "test_recall_weighted": float(rec),
            "test_f1_weighted": float(f1),
            "test_roc_auc_ovr": float(roc_auc) if not np.isnan(roc_auc) else None
        })

        # Classification report artifact
        class_report = classification_report(y_test_labels, y_pred_labels, digits=4)
        cr_path = "classification_report_xgb.txt"
        with open(cr_path, "w") as f:
            f.write("Label Mapping:\n")
            f.write(str(dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))) + "\n\n")
            f.write(class_report)
        mlflow.log_artifact(cr_path)
        try:
            os.remove(cr_path)
        except Exception:
            pass

        # Confusion matrix artifact
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_enc.classes_)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix - XGBoost Classifier")
        cm_path = "confusion_matrix_xgb.png"
        fig_cm.tight_layout()
        fig_cm.savefig(cm_path, dpi=150)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path)
        try:
            os.remove(cm_path)
        except Exception:
            pass

        # Feature importance artifact (safe extraction)
        try:
            # assemble feature names (numerical + onehot output)
            onehot = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = list(onehot.get_feature_names_out(categorical_cols))
            feature_names = numerical_cols + cat_feature_names
        except Exception:
            # fallback: use numerical_cols only
            feature_names = numerical_cols

        try:
            importances = best_model.named_steps['classifier'].feature_importances_
            feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)
            fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax_fi)
            ax_fi.set_title("Top 20 Feature Importances")
            fig_fi.tight_layout()
            fi_path = "feature_importance_xgb.png"
            fig_fi.savefig(fi_path, dpi=150)
            plt.close(fig_fi)
            mlflow.log_artifact(fi_path)
            try:
                os.remove(fi_path)
            except Exception:
                pass
        except Exception:
            # if feature importances not available or mismatch, skip
            pass

        # ------------------------
        # Save model locally & log as artifact (DagsHub-compatible)
        # ------------------------
        # Build a model signature from sample inputs -> predictions
        try:
            sample_X = X_train.head(5)
            preds_sample = best_model.predict(sample_X)
            signature = infer_signature(sample_X, preds_sample)
        except Exception:
            signature = None

        model_dir = "xgb_best_model"
        # save_model writes a directory
        mlflow.sklearn.save_model(
            sk_model=best_model,
            path=model_dir,
            signature=signature,
            input_example=X_test.head(2)
        )

        # log entire model directory as artifacts
        mlflow.log_artifacts(model_dir, artifact_path="xgb_classifier_model")
        # optionally clean up local model dir if you want
        try:
            import shutil
            shutil.rmtree(model_dir)
        except Exception:
            pass

        # final tags & summary prints
        mlflow.set_tag("total_candidates", len(cv_results))
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/emi-eligibility-mlflow.mlflow")

        print("\n=== Classification Report ===")
        print(class_report)
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc if not np.isnan(roc_auc) else 'n/a'}")
        print(f"\n🏃 View run XGBoost_Parent_Run at: {mlflow.get_artifact_uri()}")
        print(f"🧪 View experiment at: https://dagshub.com/malaychand/emi-eligibility-mlflow.mlflow/#/experiments/1")

    print("\n🎯 All MLflow runs logged successfully to DagsHub!")

if __name__ == "__main__":
    main()
