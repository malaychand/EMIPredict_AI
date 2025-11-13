# xgboost_classification_tuning_mlflow_fixed.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

def log_plot_to_mlflow(fig, artifact_name):
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)


def main():
    # ✅ Connect MLflow (DagsHub)
    dagshub.init(repo_owner="malaychand", repo_name="EMIPredict_AI", mlflow=True)
    print("✅ Connected to DagsHub MLflow Tracking Server")

    mlflow.set_experiment("EMI_prediction_classification")

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
    label_mapping = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))

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
        n_iter=8,
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
    with mlflow.start_run(run_name="XGBoost") as parent_run:
        mlflow.set_tag("author", "Malay Chand")
        mlflow.set_tag("git_commit", "unknown")
        mlflow.set_tag("model", "XGBClassifier")
        mlflow.set_tag("search_type", "randomized")
        mlflow.set_tag("task", "classification")

        print("\n🔍 Running RandomizedSearchCV...")
        random_search.fit(X_train, y_train)

        print("\n✅ Best Parameters Found:")
        print(random_search.best_params_)

        # Log best params and CV score
        mlflow.log_params({f"best_{k}": v for k, v in random_search.best_params_.items()})
        mlflow.log_metric("best_cv_score", float(random_search.best_score_))

        # Log each candidate as nested child run
        cv_results = pd.DataFrame(random_search.cv_results_)
        for idx in range(len(cv_results)):
            with mlflow.start_run(run_name=f"candidate_{idx+1}", nested=True):
                params = {k.replace("param_", ""): cv_results.loc[idx, k]
                          for k in cv_results.columns if k.startswith("param_")}
                if params:
                    mlflow.log_params(params)
                for key in ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "rank_test_score"]:
                    if key in cv_results.columns:
                        val = cv_results.loc[idx, key]
                        if pd.notna(val):
                            mlflow.log_metric(key, float(val))

        # ------------------------
        # Best model evaluation on holdout set
        # ------------------------
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        y_test_labels = label_enc.inverse_transform(y_test)
        y_pred_labels = label_enc.inverse_transform(y_pred)

        acc = accuracy_score(y_test_labels, y_pred_labels)
        prec = precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        rec = recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        # Key Performance Metrics (no "test_" prefix)
        mlflow.log_metrics({
            "accuracy": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
            "roc_auc_ovr": float(roc_auc)
        })

        # Classification report as text artifact
        class_report = classification_report(y_test_labels, y_pred_labels, digits=4)
        mlflow.log_text(
            "Label Mapping:\n" + str(label_mapping) + "\n\n" + class_report,
            "classification_report_xgb.txt"
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_enc.classes_)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix - XGBoost Classifier")
        log_plot_to_mlflow(fig_cm, "confusion_matrix_xgb.png")

        # Feature Importance (Top 20)
        try:
            onehot = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = list(onehot.get_feature_names_out(categorical_cols))
            feature_names = numerical_cols + cat_feature_names
            importances = best_model.named_steps['classifier'].feature_importances_

            feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_imp = feat_imp.sort_values('Importance', ascending=False).head(20)

            fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax_fi)
            ax_fi.set_title("Top 20 Feature Importances - XGBoost")
            log_plot_to_mlflow(fig_fi, "feature_importance_xgb.png")
        except Exception as e:
            print("⚠️ Could not log feature importance:", str(e))

        # Save model to models/ and log
        os.makedirs("models", exist_ok=True)
        model_path = "models/xgb_classifier_model.pkl"
        import joblib
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Final tags
        mlflow.set_tag("total_candidates", len(cv_results))
        mlflow.set_tag("tracking_uri", "https://dagshub.com/malaychand/EMIPredict_AI.mlflow")

        # Print summary
        print("\n=== Classification Report ===")
        print(class_report)
        print(f"📊 Key Performance Metrics:")
        print(f"   - Accuracy:  {acc:.4f}")
        print(f"   - Precision: {prec:.4f}")
        print(f"   - Recall:    {rec:.4f}")
        print(f"   - F1 Score:  {f1:.4f}")
        print(f"   - ROC-AUC:   {roc_auc:.4f}")
        print(f"\n🔗 View on DagsHub: https://dagshub.com/malaychand/EMIPredict_AI.mlflow")
        print(f"🧾 Parent Run ID: {parent_run.info.run_id}")

    print("\n🎯 All MLflow runs logged successfully to DagsHub!")


if __name__ == "__main__":
    main()